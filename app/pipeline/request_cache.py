"""
app/pipeline/request_cache.py

Request-level response cache for DEVSCOUT learning sessions:
- Key: SHA-256 of normalized request (goal, description, project, experience_level, tech_stack).
- Persisted to JSON file under data/ (default data/responses.json, gitignored).
- Bypassed when request.refresh is True.
- Returns completed LearningResponse with from_cache=True and generated_at populated.
"""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from typing import TYPE_CHECKING

import structlog

from app.models.pipeline import LearningResponse, LearnRequest

logger = structlog.get_logger(__name__)


def compute_request_cache_key(request: LearnRequest) -> str:
    """Compute deterministic SHA-256 hash of normalized request fields."""
    exp_str = (
        request.experience_level.value
        if hasattr(request.experience_level, "value")
        else str(request.experience_level or "beginner")
    ).lower().strip()

    norm = {
        "goal": request.goal.strip().lower(),
        "description": (request.description or "").strip().lower(),
        "project": (request.project or "").strip().lower(),
        "experience_level": exp_str,
        "tech_stack": sorted(t.strip().lower() for t in (request.tech_stack or []) if t.strip()),
    }
    raw = json.dumps(norm, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def get_cached_learning_response(
    request: LearnRequest,
    file_path: str | None = None,
) -> LearningResponse | None:
    """
    Retrieve cached LearningResponse if exists on disk.
    Returns None if cache miss, corrupted, or file missing.
    Populates from_cache=True and generated_at.
    """
    if file_path is None:
        from app.config import get_settings

        file_path = get_settings().request_cache_file

    if not os.path.exists(file_path):
        return None

    key = compute_request_cache_key(request)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict) or key not in data:
            return None

        entry = data[key]
        if not isinstance(entry, dict) or "response" not in entry:
            return None

        resp = LearningResponse.model_validate(entry["response"])
        resp.from_cache = True
        gen_str = entry.get("generated_at")
        if gen_str:
            try:
                resp.generated_at = datetime.fromisoformat(gen_str)
            except Exception:
                resp.generated_at = resp.created_at
        else:
            resp.generated_at = resp.created_at

        return resp
    except Exception as exc:
        logger.warning("request_cache_read_error", key=key, error=str(exc))
        return None


def save_cached_learning_response(
    request: LearnRequest,
    response: LearningResponse,
    file_path: str | None = None,
) -> None:
    """Persist completed LearningResponse to disk under request SHA-256 key."""
    if file_path is None:
        from app.config import get_settings

        file_path = get_settings().request_cache_file

    key = compute_request_cache_key(request)
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    data: dict[str, object] = {}
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    data = loaded
        except Exception:
            data = {}

    gen_dt = response.generated_at or response.created_at
    gen_iso = gen_dt.isoformat() if hasattr(gen_dt, "isoformat") else str(gen_dt)

    # Save clean serialized response
    data[key] = {
        "generated_at": gen_iso,
        "response": response.model_dump(mode="json"),
    }

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("request_cache_saved", key=key, file=file_path)
    except Exception as exc:
        logger.warning("request_cache_write_error", key=key, error=str(exc))
