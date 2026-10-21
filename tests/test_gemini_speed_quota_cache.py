"""
tests/test_gemini_speed_quota_cache.py

Unit tests for Gemini speed, client HttpOptions, fast quota failure,
persisted research cache surviving simulated restart, config defaults,
and grounded search outcome logging.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from google.genai import errors, types

from app.config import Settings, get_settings
from app.llm.provider import (
    GeminiProvider,
    LLMProviderError,
    RateLimitError,
)
from app.llm.rate_gate import GeminiRateGate
from app.models.pipeline import (
    ExperienceLevel,
    JobStatus,
    LearnJobRecord,
    LearningResponse,
    LearnRequest,
    Resource,
    ResourceType,
)
from app.pipeline.job_store import JobStore
from app.pipeline.orchestrator import run_learning_pipeline
from app.research.cache import InMemoryResearchCache
from app.research.discovery import discover_sources
from app.research.models import DiscoveredSource, ResearchDocument


# ── Test 1: Quota Error Fails in Under 2s with Friendly Message ─────────────────

@pytest.mark.asyncio
async def test_quota_error_fails_in_under_2s_with_friendly_message():
    """
    When Gemini encounters a quota exhaustion or rate limit (HTTP 429):
    1. GeminiProvider fails immediately in < 2 seconds (attempts=1, no minute-long retries).
    2. Background pipeline job transitions to FAILED with friendly error message:
       'Model quota exceeded. Try again later or check the Gemini plan.' (no internals).
    3. LearningResponse has a warning for PARTIAL_RESULTS.
    """
    mock_client = MagicMock()
    api_err = errors.APIError(429, {"error": {"message": "Quota exceeded for quota metric 'Queries' and limit 'QUOTA_EXCEEDED'", "status": "RESOURCE_EXHAUSTED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=api_err)

    provider = GeminiProvider(
        model_name="gemini-2.5-flash",
        api_key="test-api-key",
        client=mock_client,
        timeout=60.0,
    )

    # 1. Verify provider call fails fast in under 2 seconds
    t0 = time.perf_counter()
    with pytest.raises(RateLimitError):
        await provider.generate("test prompt")
    duration = time.perf_counter() - t0
    assert duration < 2.0, f"Provider took {duration:.2f}s to fail on quota error, expected < 2s"

    # 2. Verify background pipeline job error handling
    from app.api.learn import _run_pipeline_job
    store = JobStore()
    job = await store.create_job()

    async def quota_pipeline(*args, **kwargs):
        raise RateLimitError("Gemini API rate limit exceeded: Resource exhausted (429)")

    with patch("app.api.learn.get_job_store", return_value=store), patch(
        "app.api.learn.run_learning_pipeline", side_effect=quota_pipeline
    ):
        await _run_pipeline_job(
            job.job_id,
            LearnRequest(goal="Learn FastAPI background jobs", tech_stack=["fastapi"]),
        )

    updated_job = await store.get_job(job.job_id)
    assert updated_job is not None
    assert updated_job.status == JobStatus.FAILED
    assert updated_job.error == "Model quota exceeded. Try again later or check the Gemini plan."
    assert "resource exhausted" not in updated_job.error.lower()
    assert "internals" not in updated_job.error.lower()

    # 3. Check LearningResponse warning for partial results
    assert updated_job.result is not None
    warning_codes = [w.code for w in updated_job.result.warnings]
    assert "PARTIAL_RESULTS" in warning_codes


# ── Test 2: Research Cache Survives Simulated Restart ──────────────────────────

@pytest.mark.asyncio
async def test_research_cache_survives_simulated_restart(tmp_path):
    """
    Research cache entries (DiscoveredSource list and ResearchDocument):
    - Persist to JSON file under data/ (gitignored).
    - Survive server restarts (loading into a new cache instance).
    - Keys stay normalized query/URL.
    - Expired entries are discarded according to TTL.
    """
    cache_file = str(tmp_path / "data" / "test_research_cache.json")
    cache1 = InMemoryResearchCache(file_path=cache_file)

    # Store discovered sources
    sources = [
        DiscoveredSource(
            url="https://fastapi.tiangolo.com/tutorial/",
            title="FastAPI Tutorial",
            domain="fastapi.tiangolo.com",
            source_type_hint=ResourceType.DOCS,
        ),
        DiscoveredSource(
            url="https://react.dev/learn",
            title="Quick Start - React",
            domain="react.dev",
            source_type_hint=ResourceType.DOCS,
        ),
    ]
    await cache1.set("discover:fastapi tutorial", sources, ttl_seconds=3600)

    # Store a fetched research document
    doc = ResearchDocument(
        resource=Resource(
            url="https://fastapi.tiangolo.com/tutorial/",
            type=ResourceType.DOCS,
            title="FastAPI Tutorial",
            excerpt="FastAPI framework, high performance, easy to learn.",
            content_hash="hash_fastapi_123",
            last_updated=datetime.now(UTC),
        ),
        text="FastAPI framework, high performance, easy to learn.",
    )
    await cache1.set("page:https://fastapi.tiangolo.com/tutorial/", doc, ttl_seconds=3600)

    # Store an entry with very short TTL to verify expiration across restart
    await cache1.set("discover:short-lived", [{"data": "temporary"}], ttl_seconds=0.05)

    # Verify JSON file was written to disk
    assert os.path.exists(cache_file)
    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "discover:fastapi tutorial" in data
    assert "page:https://fastapi.tiangolo.com/tutorial/" in data

    # Wait for the short-lived key to expire
    await asyncio.sleep(0.08)

    # Simulate server restart by creating a completely fresh cache instance pointing to the same file
    cache2 = InMemoryResearchCache(file_path=cache_file)

    # Verify discovered sources survived and were deserialized properly
    cached_sources = await cache2.get("discover:fastapi tutorial")
    assert cached_sources is not None
    assert len(cached_sources) == 2
    assert isinstance(cached_sources[0], DiscoveredSource)
    assert cached_sources[0].url == "https://fastapi.tiangolo.com/tutorial/"
    assert cached_sources[0].source_type_hint == ResourceType.DOCS

    # Verify fetched document survived and was deserialized properly
    cached_doc = await cache2.get("page:https://fastapi.tiangolo.com/tutorial/")
    assert cached_doc is not None
    assert isinstance(cached_doc, ResearchDocument)
    assert cached_doc.resource.url == "https://fastapi.tiangolo.com/tutorial/"
    assert "high performance" in cached_doc.text

    # Verify expired entry was dropped
    assert await cache2.get("discover:short-lived") is None


# ── Test 3: Config Defaults Applied ───────────────────────────────────────────

def test_config_defaults_applied(monkeypatch):
    """
    Settings defaults:
    - max_topics_per_run == 2
    - max_evidence_retry_loops == 1
    - enable_github_queries == False
    - enable_community_queries == False
    - gemini_timeout_seconds == 60.0
    """
    monkeypatch.delenv("MAX_TOPICS_PER_RUN", raising=False)
    monkeypatch.delenv("MAX_EVIDENCE_RETRY_LOOPS", raising=False)
    monkeypatch.delenv("ENABLE_GITHUB_QUERIES", raising=False)
    monkeypatch.delenv("ENABLE_COMMUNITY_QUERIES", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT_SECONDS", raising=False)
    monkeypatch.delenv("GEMINI_TIMEOUT", raising=False)

    s = Settings(_env_file=None)
    assert s.max_topics_per_run == 2
    assert s.max_evidence_retry_loops == 1
    assert s.enable_github_queries is False
    assert s.enable_community_queries is False
    assert s.gemini_timeout_seconds == 60.0
    assert s.gemini_timeout == 60.0


@pytest.mark.asyncio
async def test_stage3_single_query_per_run_by_default(monkeypatch):
    """Stage 3 executes a single grounded query per run (docs) by default."""
    from app.models.pipeline import IntentPlanResult, LearningPlan, Milestone
    from app.pipeline import stage3_research

    recorded_queries: list[str] = []

    async def fake_discover(query, max_results=8, provider=None, http_client=None):
        recorded_queries.append(query)
        return []

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", fake_discover)

    plan_result = IntentPlanResult(
        plan=LearningPlan(
            goal_id="test-goal",
            milestones=[
                Milestone(title="FastAPI Routing", description="Learn routing basics", target_outcome="Working endpoints"),
                Milestone(title="Pydantic Schemas", description="Learn pydantic schemas", target_outcome="Validated models"),
            ],
        )
    )

    # Default run: should only execute 1 query total (tech_stack + goal + topics)
    await stage3_research.run(
        plan_result=plan_result,
        tech_stack=["FastAPI", "Python"],
    )

    assert len(recorded_queries) == 1
    q = recorded_queries[0]
    assert "FastAPI" in q
    assert "Python" in q
    assert "FastAPI Routing" in q
    assert "Pydantic Schemas" in q

    # When optional queries are enabled, 3 queries per topic should be generated
    recorded_queries.clear()
    await stage3_research.run(
        plan_result=plan_result,
        tech_stack=["FastAPI", "Python"],
        enable_github_queries=True,
        enable_community_queries=True,
    )
    assert len(recorded_queries) == 6


# ── Test 4: GeminiProvider HttpOptions and 5xx / 429 Retry Behavior ─────────────

@pytest.mark.asyncio
async def test_gemini_provider_client_http_options(monkeypatch):
    """GeminiProvider configures types.HttpOptions with timeout in ms and attempts=1."""
    captured_kwargs = {}

    def mock_client_init(**kwargs):
        captured_kwargs.update(kwargs)
        mock = MagicMock()
        mock.aio = MagicMock()
        return mock

    monkeypatch.setattr("google.genai.Client", mock_client_init)

    provider = GeminiProvider(
        model_name="gemini-2.5-flash",
        api_key="test-api-key",
        timeout=45.0,
    )
    client = provider.get_client()
    assert "http_options" in captured_kwargs
    http_opts = captured_kwargs["http_options"]
    assert isinstance(http_opts, types.HttpOptions)
    assert http_opts.timeout == 45000
    assert isinstance(http_opts.retry_options, types.HttpRetryOptions)
    assert http_opts.retry_options.attempts == 1


@pytest.mark.asyncio
async def test_gemini_provider_5xx_backoff_and_no_429_retry(tmp_path):
    """
    GeminiProvider:
    - Retries transient 5xx up to 2 times with backoff.
    - No automatic retries on 429 or quota errors (fails immediately).
    """
    mock_client = MagicMock()

    # 1. Transient 5xx succeeds on 3rd attempt
    call_count = 0
    async def flaky_5xx(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise errors.ServerError(503, {"error": {"message": "Service Unavailable"}})
        resp = MagicMock()
        resp.text = "Success after 5xx"
        resp.usage_metadata = None
        return resp

    mock_client.aio.models.generate_content = flaky_5xx
    quota_file = str(tmp_path / "test_quota.json")
    gate = GeminiRateGate(rpm_limit=100, rpd_limit=100, quota_file=quota_file)
    provider = GeminiProvider(
        model_name="gemini-2.5-flash",
        api_key="test-key",
        client=mock_client,
        rate_gate=gate,
        timeout=10.0,
    )

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await provider.generate("hello")
        assert res == "Success after 5xx"
        assert call_count == 3

    # 2. 429 is not retried: fails immediately on 1st attempt
    call_count_429 = 0
    async def rate_limited_call(*args, **kwargs):
        nonlocal call_count_429
        call_count_429 += 1
        resp_mock = MagicMock()
        resp_mock.headers = {"retry-after": "2.0"}
        resp_mock.status_code = 429
        raise errors.ClientError(429, {"error": {"message": "Rate limited"}}, resp_mock)

    mock_client.aio.models.generate_content = rate_limited_call
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        with pytest.raises(RateLimitError):
            await provider.generate("hello")
        assert call_count_429 == 1
        mock_sleep.assert_not_called()


# ── Test 5: Grounded Search Logging at INFO with Outcome ────────────────────────

@pytest.mark.asyncio
async def test_grounded_search_logging_outcome(monkeypatch):
    """Each grounded search logs at INFO with outcome ('ok' / 'quota' / 'error')."""
    logs: list[dict] = []

    def mock_info(event, **kwargs):
        logs.append({"event": event, **kwargs})

    monkeypatch.setattr("app.research.discovery.logger.info", mock_info)

    provider = AsyncMock(spec=GeminiProvider)

    # 1. Outcome 'ok'
    chunk = types.GroundingChunk(
        web=types.GroundingChunkWeb(
            uri="https://fastapi.tiangolo.com/tutorial/",
            title="FastAPI",
            domain="fastapi.tiangolo.com",
        )
    )
    provider.generate_with_search.return_value = types.GenerateContentResponse(
        candidates=[types.Candidate(grounding_metadata=types.GroundingMetadata(grounding_chunks=[chunk]))]
    )

    async def fake_resolve(url, max_redirects=3, client=None):
        return url

    monkeypatch.setattr("app.research.discovery.resolve_redirects", fake_resolve)

    # Use a dummy cache to avoid global cache hits
    dummy_cache = InMemoryResearchCache(file_path="")
    dummy_cache._entries.clear()

    await discover_sources("FastAPI routing", provider=provider, cache=dummy_cache)
    ok_logs = [l for l in logs if l.get("event") == "grounded_search" and l.get("outcome") == "ok"]
    assert len(ok_logs) == 1
    assert ok_logs[0]["query"] == "FastAPI routing"

    # 2. Outcome 'quota'
    logs.clear()
    provider.generate_with_search.side_effect = RateLimitError("Quota depleted")
    with pytest.raises(RateLimitError):
        await discover_sources("Quota test query", provider=provider, cache=dummy_cache)
    quota_logs = [l for l in logs if l.get("event") == "grounded_search" and l.get("outcome") == "quota"]
    assert len(quota_logs) == 1
    assert quota_logs[0]["query"] == "Quota test query"

    # 3. Outcome 'error'
    logs.clear()
    provider.generate_with_search.side_effect = RuntimeError("Unknown network failure")
    with pytest.raises(RuntimeError):
        await discover_sources("Error test query", provider=provider, cache=dummy_cache)
    error_logs = [l for l in logs if l.get("event") == "grounded_search" and l.get("outcome") == "error"]
    assert len(error_logs) == 1
    assert error_logs[0]["query"] == "Error test query"
