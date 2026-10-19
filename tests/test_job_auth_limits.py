"""
tests/test_job_auth_limits.py

Unit and integration tests for:
1. Asynchronous background job lifecycle and per-stage progress.
2. Pipeline overall timeout handling.
3. API key authentication on both routers (health & learn).
4. Per-key rate limiting (token bucket) on both routers.
5. Request size limits and request model field caps (goal, description).
6. SessionStore TTL eviction and max-sessions cap.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.auth import get_rate_limiter
from app.config import get_settings
from app.main import app
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ExperienceLevel,
    JobStatus,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
)
from app.pipeline.job_store import get_job_store
from app.pipeline.session_store import (
    InMemorySessionStore,
    SessionData,
    create_session_from_response,
)


@pytest.fixture(autouse=True)
async def reset_state(monkeypatch):
    """Ensure clean state for every test."""
    await get_job_store().clear()
    await get_rate_limiter().clear()
    cfg = get_settings()
    monkeypatch.setattr(cfg, "api_keys", "")
    monkeypatch.setattr(cfg, "pipeline_timeout_seconds", 300.0)
    monkeypatch.setattr(cfg, "rate_limit_capacity", 100.0)
    monkeypatch.setattr(cfg, "rate_limit_refill_rate", 50.0)
    monkeypatch.setattr(cfg, "max_request_size_bytes", 1_048_576)
    monkeypatch.setattr(cfg, "session_ttl_seconds", 86400.0)
    monkeypatch.setattr(cfg, "max_sessions", 1000)
    yield
    await get_job_store().clear()
    await get_rate_limiter().clear()


def _dummy_learning_response() -> LearningResponse:
    return LearningResponse(
        learner_profile=LearnerProfile(),
        learning_plan=LearningPlan(
            goal_id="g1",
            milestones=[
                Milestone(
                    milestone_id="m1",
                    title="Milestone 1",
                    description="M1 desc",
                    target_outcome="Understand basics",
                )
            ],
        ),
        code_gate_decision=CodeGateDecision(
            level=CodeGateLevel.EXPLAIN,
            reasons=["Explain concepts first"],
        ),
    )





# ── Test 1: Job Lifecycle & Progress ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_lifecycle_and_progress(monkeypatch):
    """POST /api/learn returns 202 immediately, reports progress, and completes with result."""
    async def fast_pipeline(request, on_progress=None, **kwargs):
        if on_progress:
            await on_progress(1, "Stage 1", 0.2)
            await asyncio.sleep(0.01)
            await on_progress(5, "Stage 5", 0.6)
            await asyncio.sleep(0.01)
            await on_progress(9, "Stage 9", 1.0)
        return _dummy_learning_response()

    monkeypatch.setattr("app.api.learn.run_learning_pipeline", fast_pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/learn", json={"goal": "Learn FastAPI background jobs"})
        assert resp.status_code == 202
        job_data = resp.json()
        job_id = job_data["job_id"]
        assert job_data["status"] == "pending"

        # Poll until complete
        final_job = None
        for _ in range(50):
            poll = await client.get(f"/api/learn/{job_id}")
            assert poll.status_code == 200
            pj = poll.json()
            if pj["status"] == "completed":
                final_job = pj
                break
            await asyncio.sleep(0.02)

        assert final_job is not None
        assert final_job["status"] == "completed"
        assert final_job["progress"] == 1.0
        assert final_job["result"]["learning_plan"]["goal_id"] == "g1"
        assert len(final_job["result"]["learning_plan"]["milestones"]) == 1

        # Check 404 for unknown job
        missing = await client.get("/api/learn/non-existent-job-uuid")
        assert missing.status_code == 404
        assert "not found" in missing.json()["detail"].lower()


# ── Test 2: Pipeline Timeout ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_job_timeout_handling(monkeypatch):
    """Job transitions to TIMED_OUT when pipeline duration exceeds pipeline_timeout_seconds."""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "pipeline_timeout_seconds", 0.05)

    async def slow_pipeline(*args, **kwargs):
        await asyncio.sleep(0.5)
        return _dummy_learning_response()

    monkeypatch.setattr("app.api.learn.run_learning_pipeline", slow_pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/api/learn", json={"goal": "Timeout test pipeline"})
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        final_job = None
        for _ in range(50):
            poll = await client.get(f"/api/learn/{job_id}")
            pj = poll.json()
            if pj["status"] in ("timed_out", "failed", "completed"):
                final_job = pj
                break
            await asyncio.sleep(0.02)

        assert final_job is not None
        assert final_job["status"] == "timed_out"
        assert "timed out" in final_job["error"].lower()


# ── Test 3: API-Key Auth on Both Routers ──────────────────────────────────────

@pytest.mark.asyncio
async def test_api_key_auth_enforced_on_both_routers(monkeypatch):
    """When api_keys is set, missing or invalid key returns 401 on health and learn routers."""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "api_keys", "key-alpha, key-beta")

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Health router: no key -> 401
        h_no_key = await client.get("/health")
        assert h_no_key.status_code == 401

        # Health router: invalid key -> 401
        h_bad_key = await client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert h_bad_key.status_code == 401

        # Health router: valid X-API-Key -> 200
        h_good_key = await client.get("/health", headers={"X-API-Key": "key-alpha"})
        assert h_good_key.status_code == 200

        # Health router: valid Bearer token -> 200
        h_bearer = await client.get("/health", headers={"Authorization": "Bearer key-beta"})
        assert h_bearer.status_code == 200

        # Learn router: no key -> 401
        l_no_key = await client.post("/api/learn", json={"goal": "Auth test"})
        assert l_no_key.status_code == 401

        # Learn router: valid key -> 202
        async def mock_pipe(*args, **kwargs):
            return _dummy_learning_response()

        monkeypatch.setattr("app.api.learn.run_learning_pipeline", mock_pipe)
        l_good_key = await client.post(
            "/api/learn",
            json={"goal": "Auth test"},
            headers={"X-API-Key": "key-alpha"},
        )
        assert l_good_key.status_code == 202


# ── Test 4: Per-Key Rate Limiting on Both Routers ─────────────────────────────

@pytest.mark.asyncio
async def test_rate_limiting_token_bucket(monkeypatch):
    """Exceeding capacity returns 429, tokens isolate across keys."""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "api_keys", "client-1, client-2")
    monkeypatch.setattr(cfg, "rate_limit_capacity", 2.0)
    monkeypatch.setattr(cfg, "rate_limit_refill_rate", 0.01)  # slow refill

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        h1 = {"X-API-Key": "client-1"}
        h2 = {"X-API-Key": "client-2"}

        # Client 1: 2 tokens allowed
        r1 = await client.get("/health", headers=h1)
        assert r1.status_code == 200
        r2 = await client.get("/health", headers=h1)
        assert r2.status_code == 200

        # Client 1: 3rd request -> 429
        r3 = await client.get("/health", headers=h1)
        assert r3.status_code == 429
        assert "rate limit exceeded" in r3.json()["detail"].lower()

        # Client 2 has its own separate bucket -> 200
        r4 = await client.get("/health", headers=h2)
        assert r4.status_code == 200


# ── Test 5: Request Size Limits & Field Length Caps ───────────────────────────

@pytest.mark.asyncio
async def test_request_size_limits_and_field_caps(monkeypatch):
    """Content-Length exceeding max returns 413; oversized goal/description returns 422."""
    cfg = get_settings()
    monkeypatch.setattr(cfg, "max_request_size_bytes", 500)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Payload > 500 bytes -> 413
        large_body = "x" * 600
        resp_413 = await client.post(
            "/api/learn",
            content=large_body,
            headers={"Content-Length": str(len(large_body)), "Content-Type": "application/json"},
        )
        assert resp_413.status_code == 413
        assert "too large" in resp_413.json()["detail"].lower()

    # Reset size cap for field validations
    monkeypatch.setattr(cfg, "max_request_size_bytes", 1_048_576)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 2. Goal length > 1000 characters -> 422
        long_goal = "a" * 1001
        resp_long_goal = await client.post("/api/learn", json={"goal": long_goal})
        assert resp_long_goal.status_code == 422

        # 3. Description length > 2000 characters -> 422
        long_desc = "b" * 2001
        resp_long_desc = await client.post(
            "/api/learn",
            json={"goal": "Valid goal title", "description": long_desc},
        )
        assert resp_long_desc.status_code == 422

        # 4. Valid lengths pass validation
        valid_goal = "c" * 500
        valid_desc = "d" * 1000
        resp_valid = await client.post(
            "/api/learn",
            json={"goal": valid_goal, "description": valid_desc},
        )
        assert resp_valid.status_code == 202


# ── Test 6: SessionStore TTL & Max-Sessions Eviction ──────────────────────────

@pytest.mark.asyncio
async def test_session_store_ttl_and_max_sessions_eviction():
    """SessionStore evicts expired items on access and oldest items on capacity breach."""
    resp = _dummy_learning_response()

    # 1. TTL eviction
    short_ttl_store = InMemorySessionStore(ttl_seconds=0.05, max_sessions=10)
    s1 = create_session_from_response(resp)
    s1.response_id = "sess-ttl"
    await short_ttl_store.save(s1)

    # Immediately present
    found = await short_ttl_store.get("sess-ttl")
    assert found is not None

    # Wait for TTL expiry
    await asyncio.sleep(0.08)
    expired = await short_ttl_store.get("sess-ttl")
    assert expired is None

    # 2. Max-sessions cap eviction
    cap_store = InMemorySessionStore(ttl_seconds=3600.0, max_sessions=2)
    s_a = create_session_from_response(resp)
    s_a.response_id = "sess-a"
    s_b = create_session_from_response(resp)
    s_b.response_id = "sess-b"
    s_c = create_session_from_response(resp)
    s_c.response_id = "sess-c"

    await cap_store.save(s_a)
    await asyncio.sleep(0.01)
    await cap_store.save(s_b)

    assert await cap_store.get("sess-a") is not None
    assert await cap_store.get("sess-b") is not None

    # Save 3rd session; oldest (sess-a) must be evicted
    await asyncio.sleep(0.01)
    await cap_store.save(s_c)

    assert await cap_store.get("sess-a") is None
    assert await cap_store.get("sess-b") is not None
    assert await cap_store.get("sess-c") is not None
