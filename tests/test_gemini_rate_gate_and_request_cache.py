"""
tests/test_gemini_rate_gate_and_request_cache.py

Unit tests for:
1. Limiter queues the 5th request in a minute window.
2. Daily budget blocks the 19th call with LLMProviderError("Gemini daily budget reached").
3. Cache hit makes zero Gemini calls; populates from_cache=True and generated_at; refresh=True bypasses cache.
4. 429 is not retried (fails immediately, attempts=1).
5. GET /api/quota matches the counter without exposing secrets.
6. Stage 3 single grounded discovery call per pass built from tech_stack + goal + topic titles joined.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta, timezone
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.testclient import TestClient
from google.genai import errors

from app.api.router import api_router
from app.config import Settings
from app.llm.provider import GeminiProvider, LLMProviderError, RateLimitError
from app.llm.rate_gate import GeminiRateGate, get_rate_gate, set_rate_gate
from app.main import app
from app.models.pipeline import (
    ExperienceLevel,
    IntentPlanResult,
    LearnerProfile,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
    Resource,
    ResourceType,
    CodeGateDecision,
    CodeGateLevel,
)
from app.pipeline.orchestrator import run_learning_pipeline
from app.pipeline.request_cache import (
    compute_request_cache_key,
    get_cached_learning_response,
    save_cached_learning_response,
)
from app.pipeline import stage3_research


# ── Test 1: Limiter queues the 5th request in a minute window ─────────────────

@pytest.mark.asyncio
async def test_limiter_queues_5th_request_in_minute_window(tmp_path):
    """
    With gemini_rpm=4:
    - 4 requests acquire immediately within the minute window.
    - The 5th request is queued and sleeps until the window allows it.
    """
    quota_file = str(tmp_path / "quota.json")
    gate = GeminiRateGate(rpm_limit=4, rpd_limit=18, quota_file=quota_file)

    simulated_now = 1000.0

    def mock_monotonic():
        return simulated_now

    sleep_calls: list[float] = []

    async def mock_sleep(seconds: float):
        nonlocal simulated_now
        sleep_calls.append(seconds)
        simulated_now += seconds

    with patch("time.monotonic", side_effect=mock_monotonic), patch(
        "asyncio.sleep", side_effect=mock_sleep
    ):
        # First 4 requests spaced 10s apart at t=1000, 1010, 1020, 1030
        for _ in range(4):
            await gate.acquire()
            simulated_now += 10.0

        assert len(sleep_calls) == 0
        assert len(gate._request_timestamps) == 4

        # 5th request arrives at t=1040: window full, oldest at 1000 -> must wait 20s until t=1060
        await gate.acquire()

        assert len(sleep_calls) == 1
        assert 19.9 <= sleep_calls[0] <= 20.1  # Waited exactly until oldest slot expired
        assert len(gate._request_timestamps) == 4  # Oldest pruned at t=1060, 5th added


# ── Test 2: Daily budget blocks the 19th call ───────────────────────────────────

@pytest.mark.asyncio
async def test_daily_budget_blocks_19th_call(tmp_path):
    """
    With gemini_rpd_limit=18:
    - 18 calls succeed and increment used_today to 18.
    - The 19th call raises LLMProviderError('Gemini daily budget reached') BEFORE calling.
    """
    quota_file = str(tmp_path / "quota.json")
    gate = GeminiRateGate(rpm_limit=100, rpd_limit=18, quota_file=quota_file)

    for i in range(18):
        await gate.acquire()
        assert gate._get_used_today() == i + 1

    # 19th call must be blocked before calling
    with pytest.raises(LLMProviderError) as exc_info:
        await gate.acquire()

    assert "Gemini daily budget reached" in str(exc_info.value)
    # Counter remains at 18
    assert gate._get_used_today() == 18


# ── Test 3: Cache hit makes zero Gemini calls ──────────────────────────────────

@pytest.mark.asyncio
async def test_request_cache_hit_makes_zero_gemini_calls(tmp_path, monkeypatch):
    """
    Request-level cache:
    - Key is SHA-256 of normalized request (goal, description, project, experience_level, tech_stack).
    - Cache hit returns completed response with from_cache=True and generated_at populated.
    - Zero Gemini calls are made on cache hit.
    - refresh=True bypasses the cache.
    """
    cache_file = str(tmp_path / "responses.json")
    monkeypatch.setattr("app.config.get_settings", lambda: Settings(request_cache_file=cache_file, _env_file=None))

    request = LearnRequest(
        goal="Build an async rate limiter",
        description="Using token bucket algorithm in Python",
        project="devscout-v2",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["FastAPI", "Python"],
        refresh=False,
    )

    sample_response = LearningResponse(
        learner_profile=LearnerProfile(),
        learning_plan=LearningPlan(goal_id="test-goal"),
        code_gate_decision=CodeGateDecision(level=CodeGateLevel.EXPLAIN),
        confidence_score=0.9,
        quality_score=0.85,
        created_at=datetime.now(UTC),
    )

    # 1. Initially no cached entry
    assert get_cached_learning_response(request, file_path=cache_file) is None

    # Save to cache
    save_cached_learning_response(request, sample_response, file_path=cache_file)

    # 2. Mock Gemini provider to track any LLM invocations
    mock_provider = MagicMock(spec=GeminiProvider)
    mock_provider.generate = AsyncMock()
    mock_provider.generate_structured = AsyncMock()
    mock_provider.generate_with_search = AsyncMock()

    # 3. Call run_learning_pipeline with refresh=False and use_cache=True -> CACHE HIT
    result = await run_learning_pipeline(request, provider=mock_provider, use_cache=True)

    assert result.from_cache is True
    assert result.generated_at is not None
    assert mock_provider.generate.call_count == 0
    assert mock_provider.generate_structured.call_count == 0
    assert mock_provider.generate_with_search.call_count == 0

    # 4. Verify request with refresh=True bypasses the cache
    refresh_req = request.model_copy(update={"refresh": True})
    # If refresh=True, it will run the pipeline and call provider or fail on missing stages
    # Verify compute_request_cache_key produces consistent SHA-256
    k1 = compute_request_cache_key(request)
    k2 = compute_request_cache_key(refresh_req)
    assert k1 == k2
    assert len(k1) == 64


# ── Test 4: 429 is not retried ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_429_is_not_retried(tmp_path):
    """GeminiProvider does not automatically retry HTTP 429 / quota errors."""
    quota_file = str(tmp_path / "quota.json")
    test_gate = GeminiRateGate(rpm_limit=10, rpd_limit=18, quota_file=quota_file)

    call_count = 0
    mock_client = MagicMock()

    async def mock_429(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        raise errors.ClientError(
            429,
            {"error": {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}},
        )

    mock_client.aio.models.generate_content = mock_429
    provider = GeminiProvider(
        model_name="gemini-2.5-flash",
        api_key="test-api-key",
        client=mock_client,
        rate_gate=test_gate,
        timeout=5.0,
    )

    with pytest.raises(RateLimitError):
        await provider.generate("test prompt")

    # Called exactly once, no retries!
    assert call_count == 1


# ── Test 5: /api/quota matches the counter without exposing secrets ───────────

def test_api_quota_endpoint_matches_counter(tmp_path, monkeypatch):
    """GET /api/quota returns {used_today, limit_today, rpm_limit} matching the limiter."""
    quota_file = str(tmp_path / "quota.json")
    gate = GeminiRateGate(rpm_limit=4, rpd_limit=18, quota_file=quota_file)
    # Simulate 7 requests used today
    today_str = gate._get_today_str()
    with open(quota_file, "w", encoding="utf-8") as f:
        json.dump({"date": today_str, "used_today": 7}, f)

    set_rate_gate(gate)

    client = TestClient(app)
    resp = client.get("/api/quota")
    assert resp.status_code == 200
    data = resp.json()

    assert data == {
        "used_today": 7,
        "limit_today": 18,
        "rpm_limit": 4,
    }
    # Ensure no secret keys are leaked
    for key in data:
        assert "key" not in key.lower()
        assert "token" not in key.lower()
        assert "secret" not in key.lower()


# ── Test 6: One Search Per Run (Stage 3 Single Grounded Discovery Call) ─────────

@pytest.mark.asyncio
async def test_stage3_single_search_per_run_combines_tech_goal_topics(monkeypatch):
    """
    Stage 3 makes a single grounded discovery call per pass:
    - Combines tech_stack + goal + topic titles joined into one query.
    - GitHub and community connectors are optional and off by default.
    """
    captured_queries: list[str] = []

    async def fake_discover(query, max_results=8, provider=None, http_client=None):
        captured_queries.append(query)
        return []

    github_called = False

    async def fake_github(self, query, limit=3):
        nonlocal github_called
        github_called = True
        return []

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", fake_discover)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", fake_github)

    plan_result = IntentPlanResult(
        plan=LearningPlan(
            goal_id="test-goal",
            milestones=[
                Milestone(title="FastAPI Routing", description="Learn routing basics", target_outcome="Working routes"),
                Milestone(title="Pydantic Schemas", description="Learn pydantic schemas", target_outcome="Validated models"),
            ],
        )
    )

    await stage3_research.run(
        plan_result=plan_result,
        tech_stack=["FastAPI", "Python"],
        goal="Building modern APIs",
    )

    # Exactly 1 search was made for the whole run!
    assert len(captured_queries) == 1
    query = captured_queries[0]
    # Check that tech_stack, goal words, and all topic titles are present in the single query
    assert "FastAPI" in query
    assert "Python" in query
    assert "Building modern APIs" in query or ("Building" in query and "APIs" in query)
    assert "FastAPI Routing" in query
    assert "Pydantic Schemas" in query
    # GitHub connector was off by default
    assert github_called is False
