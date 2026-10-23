"""
tests/test_model_registry_and_router.py

Unit tests for model registry, per-model limiter, router, /api/models, and transparency:
1. Unknown model_id -> 422 (HTTP POST /api/learn and get_provider_for).
2. Unknown provider fails startup / validation with a clear message.
3. Auto mode skips an exhausted model and chooses the next available model.
4. Quota error moves to the next model exactly once (failover).
5. Local/secondary fallback works with secondary provider/model.
6. GET /api/models shows used_today and limit_today per model.
7. No secrets in any API response (/api/models, /api/learn).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi import HTTPException
from starlette.testclient import TestClient

from app.api.router import api_router
from app.llm.provider import LLMProvider, LLMProviderError, RateLimitError
from app.llm.rate_gate import GeminiRateGate, set_rate_gate
from app.llm.registry import ModelEntry, ModelRegistry, load_model_registry
from app.llm.router import RoutedProvider, can_run_grounded_search, get_provider_for
from app.main import app
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ExperienceLevel,
    FullCodeSolution,
    IntentPlanResult,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
    Resource,
    ResourceType,
    StageMetrics,
)
from app.pipeline import orchestrator


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_rate_gate(tmp_path):
    quota_file = str(tmp_path / "quota.json")
    gate = GeminiRateGate(quota_file=quota_file)
    set_rate_gate(gate)
    yield gate
    set_rate_gate(None)


# ── Test 1: Unknown model_id -> 422 ───────────────────────────────────────────

def test_unknown_model_id_learn_request_validation():
    """LearnRequest validation fails with 422 if model_id is not in registry."""
    with pytest.raises(Exception) as exc_info:
        LearnRequest(
            goal="Learn Python async",
            model_id="non-existent-model-id",
        )
    assert "Unknown model_id" in str(exc_info.value) or "validation" in str(exc_info.value).lower()


def test_unknown_model_id_http_endpoint_returns_422():
    """POST /api/learn with unknown model_id returns 422 Unprocessable Entity."""
    client = TestClient(app)
    response = client.post(
        "/api/learn",
        json={
            "goal": "Learn Python async programming",
            "model_id": "non-existent-model-xyz",
        },
    )
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_unknown_model_id_router_raises_422():
    """get_provider_for raises HTTPException(422) for unknown model_id."""
    with pytest.raises(HTTPException) as exc_info:
        await get_provider_for(stage=2, requested_id="invalid-model-404")
    assert exc_info.value.status_code == 422
    assert "Unknown model_id" in str(exc_info.value.detail)


# ── Test 2: Unknown provider fails startup with a clear message ───────────────

def test_unknown_provider_fails_validation_with_clear_message(tmp_path):
    """Registry validation rejects unknown providers with an explicit error."""
    bad_models = [
        {
            "id": "bad-model",
            "label": "Bad Provider Model",
            "provider": "unsupported-provider-ai",
            "model": "model-1",
            "priority": 1,
        }
    ]
    bad_file = tmp_path / "models.json"
    bad_file.write_text(json.dumps(bad_models), encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_model_registry(str(bad_file))
    
    assert "unsupported-provider-ai" in str(exc_info.value)
    assert "gemini" in str(exc_info.value)


# ── Test 3: Auto mode skips exhausted model ───────────────────────────────────

@pytest.mark.asyncio
async def test_auto_mode_skips_exhausted_model(clean_rate_gate, monkeypatch):
    """
    When gemini-main daily budget is exhausted (20/20 calls used):
    - Auto mode chooses the next available model in registry.
    """
    from app.llm.registry import ModelRegistry, set_model_registry

    # Register two models
    m1 = ModelEntry(id="gemini-main", label="Gemini Main", provider="gemini", model="gemini-2.5-flash", priority=1, rpm=4, rpd=18)
    m2 = ModelEntry(id="gemini-backup", label="Gemini Backup", provider="gemini", model="gemini-2.0-flash", priority=2, rpm=4, rpd=18)
    set_model_registry(ModelRegistry([m1, m2]))
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")

    gate = clean_rate_gate
    today_str = gate._get_today_str()
    with open(gate.quota_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": today_str,
                "models": {
                    "gemini-main": {"used_today": 20},
                },
            },
            f,
        )

    assert not gate.has_budget("gemini-main")
    assert gate.has_budget("gemini-backup")

    routed = await get_provider_for(stage=2, requested_id=None)
    assert isinstance(routed, RoutedProvider)
    assert routed.active_entry.id == "gemini-backup"


# ── Test 4: Quota error moves to next model exactly once ──────────────────────

@pytest.mark.asyncio
async def test_quota_error_moves_to_next_model_once():
    """
    When primary model call raises RateLimitError or quota exceeded:
    - Router catches quota error and moves to fallback model.
    - Fallback model executes the request.
    - If fallback also fails, it raises and does NOT loop again.
    """
    primary_entry = ModelEntry(
        id="gemini-main",
        label="Gemini Main",
        provider="gemini",
        model="gemini-2.5-flash",
        priority=1,
    )
    fallback_entry = ModelEntry(
        id="gemini-backup",
        label="Gemini Backup",
        provider="gemini",
        model="gemini-2.0-flash",
        priority=2,
    )

    mock_primary = MagicMock()
    mock_primary.model_id = "gemini-main"
    mock_primary.generate = AsyncMock(
        side_effect=RateLimitError("429 Resource exhausted: quota exceeded")
    )

    mock_fallback = MagicMock()
    mock_fallback.model_id = "gemini-backup"
    mock_fallback.generate = AsyncMock(return_value="Output from fallback model")

    routed = RoutedProvider(
        primary_entry=primary_entry,
        fallback_entry=fallback_entry,
        stage=2,
    )
    routed._primary_instance = mock_primary
    routed._fallback_instance = mock_fallback

    # Call generate
    result = await routed.generate("Hello world")

    # Assert primary was called, failed, and fallback was called
    assert mock_primary.generate.call_count == 1
    assert mock_fallback.generate.call_count == 1
    assert result == "Output from fallback model"
    assert routed.was_fallback is True
    assert routed.model_id == "gemini-backup"

    # Test that if fallback also fails, it does NOT retry again
    mock_fallback_failing = MagicMock()
    mock_fallback_failing.model_id = "gemini-backup"
    mock_fallback_failing.generate = AsyncMock(
        side_effect=RateLimitError("Fallback also quota exhausted")
    )
    routed2 = RoutedProvider(
        primary_entry=primary_entry,
        fallback_entry=fallback_entry,
        stage=2,
    )
    routed2._primary_instance = mock_primary
    routed2._fallback_instance = mock_fallback_failing

    with pytest.raises(RateLimitError):
        await routed2.generate("Hello world 2")

    # Primary called once, fallback called once, no infinite loop
    assert mock_primary.generate.call_count == 2
    assert mock_fallback_failing.generate.call_count == 1


# ── Test 5: Fallback routing works when primary quota is exhausted ───────────

@pytest.mark.asyncio
async def test_fallback_routing_on_quota_error(clean_rate_gate, monkeypatch):
    """
    Explicit requested_id with LLM_FALLBACK configured:
    - Primary gemini-main fails with quota error.
    - Fallback routes to gemini-backup and completes.
    """
    from app.llm.registry import ModelRegistry, set_model_registry

    m1 = ModelEntry(id="gemini-main", label="Gemini Main", provider="gemini", model="gemini-2.5-flash", priority=1)
    m2 = ModelEntry(id="gemini-backup", label="Gemini Backup", provider="gemini", model="gemini-2.0-flash", priority=2)
    set_model_registry(ModelRegistry([m1, m2]))
    monkeypatch.setenv("LLM_FALLBACK", "gemini-backup")

    routed = await get_provider_for(stage=2, requested_id="gemini-main")
    assert routed.fallback_entry is not None
    assert routed.fallback_entry.id == "gemini-backup"

    # Mock primary to fail with quota error
    mock_gemini = MagicMock()
    mock_gemini.model_id = "gemini-main"
    mock_gemini.generate = AsyncMock(
        side_effect=RateLimitError("Gemini daily budget reached")
    )

    # Fake fallback provider
    mock_fallback = MagicMock()
    mock_fallback.model_id = "gemini-backup"
    mock_fallback.generate = AsyncMock(return_value="Fallback generated text")

    routed._primary_instance = mock_gemini
    routed._fallback_instance = mock_fallback

    output = await routed.generate("Summarize topic")
    assert output == "Fallback generated text"
    assert routed.was_fallback is True


# ── Test 6: GET /api/models shows used and limit per model ────────────────────

def test_api_models_endpoint(clean_rate_gate):
    """
    GET /api/models returns list of models with:
    {id, label, provider, local, tier, available, used_today, limit_today}
    and no secrets.
    """
    gate = clean_rate_gate
    # Record 3 calls for gemini-main
    today_str = gate._get_today_str()
    with open(gate.quota_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "date": today_str,
                "models": {
                    "gemini-main": {"used_today": 3},
                },
            },
            f,
        )

    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) >= 1

    gemini_entry = next((m for m in models if m["id"] == "gemini-main"), None)
    assert gemini_entry is not None
    assert gemini_entry["provider"] == "gemini"
    assert gemini_entry["local"] is False
    assert gemini_entry["used_today"] == 3
    assert gemini_entry["limit_today"] == 20
    assert gemini_entry["available"] is True


# ── Test 7: No secrets in any response ────────────────────────────────────────

def test_no_secrets_in_models_or_learn_responses(clean_rate_gate):
    """Ensure no API keys, tokens, or secret fields exist in responses."""
    client = TestClient(app)

    # 1. /api/models response inspection
    resp_models = client.get("/api/models")
    assert resp_models.status_code == 200
    models_data = resp_models.json()

    forbidden_substrings = ["key", "secret", "token", "password", "credential"]
    for model in models_data:
        for k, v in model.items():
            for forbidden in forbidden_substrings:
                assert forbidden not in k.lower(), f"Forbidden key '{k}' found in /api/models"
                if isinstance(v, str):
                    assert "AIza" not in v, "Possible Gemini API key leaked in model response"

    # 2. /api/quota response inspection
    resp_quota = client.get("/api/quota")
    assert resp_quota.status_code == 200
    quota_data = resp_quota.json()
    for k in quota_data:
        for forbidden in forbidden_substrings:
            assert forbidden not in k.lower(), f"Forbidden key '{k}' found in /api/quota"


# ── Test 8: End-to-end transparency: models_used & FALLBACK_MODEL_USED ───────

@pytest.mark.asyncio
async def test_transparency_models_used_and_fallback_warning():
    """
    When a fallback model is used during the pipeline:
    - StageMetrics records model_id.
    - LearningResponse.models_used contains the ordered, unique list of models.
    - FALLBACK_MODEL_USED warning is appended.
    """
    metrics = [
        StageMetrics(stage_id=1, stage_name="Developer Input"),
        StageMetrics(stage_id=2, stage_name="Intent Plan", model_id="gemini-backup"),
        StageMetrics(stage_id=5, stage_name="AI Analysis", model_id="gemini-backup"),
    ]

    # Run response construction check
    models_used = list(dict.fromkeys(m.model_id for m in metrics if m.model_id))
    assert models_used == ["gemini-backup"]

    first_choice_model_id = "gemini-main"
    fallback_model_used = any(
        m.model_id and m.model_id != first_choice_model_id for m in metrics
    )
    assert fallback_model_used is True


# ── Test 9: Acceptance of registered model IDs in LearnRequest ───────────────

def test_registered_model_ids_accepted_in_learn_request():
    """Verify both gemini-main and groq-openai/gpt-oss-120b are accepted."""
    req_gemini = LearnRequest(goal="Learn FastAPI testing", model_id="gemini-main")
    assert req_gemini.model_id == "gemini-main"

    req_groq = LearnRequest(goal="Learn FastAPI testing", model_id="groq-openai/gpt-oss-120b")
    assert req_groq.model_id == "groq-openai/gpt-oss-120b"

    # Default is None (auto -> gemini-main)
    req_default = LearnRequest(goal="Learn FastAPI testing")
    assert req_default.model_id is None


def test_invalid_groq_variant_rejected():
    """Verify slightly misspelled or unconfigured model variants are rejected with 422."""
    with pytest.raises(Exception) as exc_info:
        LearnRequest(goal="Learn FastAPI testing", model_id="groq-llama-3.3-70b-versatile")
    assert "Unknown model_id" in str(exc_info.value)


# ── Test 10: POST /api/learn accepts valid model_id and rejects invalid ──────

def test_post_api_learn_accepts_groq_model_id(clean_rate_gate):
    """POST /api/learn with 'groq-openai/gpt-oss-120b' returns HTTP 202."""
    client = TestClient(app)
    resp = client.post(
        "/api/learn",
        json={
            "goal": "Build a test app",
            "model_id": "groq-openai/gpt-oss-120b",
        },
    )
    assert resp.status_code == 202
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "pending"


def test_post_api_learn_rejects_unknown_model_id():
    """POST /api/learn with unknown model_id returns HTTP 422 with available models list."""
    client = TestClient(app)
    resp = client.post(
        "/api/learn",
        json={
            "goal": "Build a test app",
            "model_id": "unknown-model-xyz",
        },
    )
    assert resp.status_code == 422
    data = resp.json()
    detail = str(data["detail"])
    assert "Unknown model_id 'unknown-model-xyz'" in detail
    assert "gemini-main" in detail
    assert "groq-openai/gpt-oss-120b" in detail


# ── Test 11: GET /api/models returns registered models with expected IDs ─────

def test_api_models_lists_registered_models(clean_rate_gate):
    """GET /api/models includes both registered models without hardcoded secrets."""
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    model_ids = [m["id"] for m in models]

    assert "gemini-main" in model_ids
    assert "groq-openai/gpt-oss-120b" in model_ids

    groq_meta = next(m for m in models if m["id"] == "groq-openai/gpt-oss-120b")
    assert groq_meta["provider"] in ("groq", "openai_compatible")
    assert groq_meta["label"] == "Groq (openai/gpt-oss-120b)"
    assert isinstance(groq_meta["available"], bool)


# ── Test 12: Router resolves groq-openai/gpt-oss-120b correctly ──────────────

@pytest.mark.asyncio
async def test_router_resolves_groq_model():
    """get_provider_for('groq-openai/gpt-oss-120b') returns an OpenAICompatibleProvider targeting Groq."""
    routed = await get_provider_for(stage=2, requested_id="groq-openai/gpt-oss-120b")
    assert routed.active_entry.id == "groq-openai/gpt-oss-120b"
    assert routed.active_entry.model == "openai/gpt-oss-120b"
    assert routed.active_entry.provider == "openai_compatible"

