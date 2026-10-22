"""
tests/test_data_dir_isolation.py

Test verifying test isolation for repo data directory:
Ensures normal requests through the app, pipeline, rate gate, request cache,
and research cache operate against tmp_path fixtures and never create or modify
the repository's data/ directory.
"""
from __future__ import annotations

import os
from pathlib import Path
import pytest
from starlette.testclient import TestClient

from app.config import get_settings
from app.llm.rate_gate import get_rate_gate
from app.main import app
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ExperienceLevel,
    LearnerProfile,
    LearningPlan,
    LearningResponse,
    LearnRequest,
)
from app.pipeline.orchestrator import run_learning_pipeline
from app.pipeline.request_cache import (
    get_cached_learning_response,
    save_cached_learning_response,
)
from app.research.cache import get_research_cache
from app.research.models import DiscoveredSource, ResourceType


def _snapshot_dir(dir_path: Path) -> dict[str, tuple[float, int]]:
    """Return map of filename -> (mtime, size) for all files in dir_path."""
    if not dir_path.exists():
        return {}
    snapshot: dict[str, tuple[float, int]] = {}
    for entry in dir_path.iterdir():
        if entry.is_file():
            stat = entry.stat()
            snapshot[entry.name] = (stat.st_mtime, stat.st_size)
    return snapshot


@pytest.mark.asyncio
async def test_normal_request_does_not_create_or_modify_repo_data_dir(tmp_path):
    """
    Runs normal requests and operations through the app and pipeline.
    Asserts the repo's data/ directory was not created or modified,
    and all persisted state is strictly confined to tmp_path.
    """
    repo_root = Path(__file__).resolve().parent.parent
    repo_data_dir = repo_root / "data"

    data_existed_before = repo_data_dir.exists()
    snapshot_before = _snapshot_dir(repo_data_dir)

    # Verify settings point inside tmp_path
    cfg = get_settings()
    assert str(tmp_path) in cfg.gemini_quota_file
    assert str(tmp_path) in cfg.request_cache_file
    assert str(tmp_path) in cfg.research_cache_file

    # 1. Normal HTTP requests through FastAPI app
    client = TestClient(app)

    # GET /api/quota
    res_quota = client.get("/api/quota")
    assert res_quota.status_code == 200

    # GET /api/models
    res_models = client.get("/api/models")
    assert res_models.status_code == 200

    # POST /api/learn
    res_learn = client.post(
        "/api/learn",
        json={
            "goal": "Verify Repo Data Isolation",
            "experience_level": "intermediate",
            "tech_stack": ["Python"],
        },
    )
    assert res_learn.status_code == 202

    # 2. Rate gate operations (must write to tmp_path quota file)
    gate = get_rate_gate()
    await gate.acquire("gemini-main")
    assert gate._get_used_today("gemini-main") >= 1
    assert os.path.exists(cfg.gemini_quota_file)
    assert str(tmp_path) in gate.quota_file

    # 3. Request cache save & load (must write to tmp_path responses file)
    req = LearnRequest(
        goal="Build an isolated component",
        experience_level=ExperienceLevel.BEGINNER,
        tech_stack=["Python"],
    )
    dummy_response = LearningResponse(
        learner_profile=LearnerProfile(),
        learning_plan=LearningPlan(goal_id="isolated-goal"),
        code_gate_decision=CodeGateDecision(level=CodeGateLevel.EXPLAIN),
        confidence_score=1.0,
        quality_score=0.9,
    )
    save_cached_learning_response(req, dummy_response)
    cached = get_cached_learning_response(req)
    assert cached is not None
    assert cached.from_cache is True
    assert os.path.exists(cfg.request_cache_file)

    # 4. Research cache operations (must write to tmp_path research cache file)
    res_cache = get_research_cache()
    await res_cache.set(
        "discover:test",
        [DiscoveredSource(url="https://example.com/docs", title="Example Docs", domain="example.com", source_type_hint=ResourceType.DOCS)],
    )
    loaded_sources = await res_cache.get("discover:test")
    assert loaded_sources is not None
    assert os.path.exists(cfg.research_cache_file)

    # 5. Snapshot assertion on repo data/ directory
    if not data_existed_before:
        assert not repo_data_dir.exists(), "repo data/ directory was created during test run!"
    else:
        snapshot_after = _snapshot_dir(repo_data_dir)
        assert snapshot_after == snapshot_before, (
            f"repo data/ directory was modified during test run!\n"
            f"Before: {snapshot_before}\n"
            f"After:  {snapshot_after}"
        )
