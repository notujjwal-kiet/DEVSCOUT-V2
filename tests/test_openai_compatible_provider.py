"""
tests/test_openai_compatible_provider.py

Unit tests for openai_compatible provider kind:
1. key missing -> unavailable and never selected in auto mode;
2. bad base_url rejected at startup;
3. 429 not retried and moves on once;
4. key never present in logs, errors, or API responses;
5. JSON-mode parsing with repair;
6. usage captured into StageMetrics with per-model registry pricing.
All tests use httpx.MockTransport (zero network calls).
"""
from __future__ import annotations

import json
import os
import pytest
import httpx
from pydantic import BaseModel
from starlette.testclient import TestClient

from app.llm.provider import (
    LLMProviderError,
    OpenAICompatibleProvider,
    RateLimitError,
    get_llm_provider,
)
from app.llm.rate_gate import GeminiRateGate, set_rate_gate
from app.llm.registry import (
    ModelEntry,
    ModelRegistry,
    is_model_available,
    load_model_registry,
    set_model_registry,
)
from app.llm.router import RoutedProvider, get_provider_for
from app.main import app
from app.models.pipeline import StageMetrics


@pytest.fixture(autouse=True)
def reset_registry():
    yield
    set_model_registry(None)


@pytest.fixture
def clean_rate_gate(tmp_path):
    quota_file = str(tmp_path / "quota.json")
    gate = GeminiRateGate(quota_file=quota_file)
    set_rate_gate(gate)
    yield gate
    set_rate_gate(None)



# ── Test 1: Bad base_url rejected at startup ─────────────────────────────────

def test_bad_base_url_rejected_at_startup(tmp_path):
    """base_url must use HTTPS (HTTP is allowed only for localhost entries)."""
    # 1. Plain HTTP to external domain is rejected
    with pytest.raises(ValueError) as exc:
        ModelEntry(
            id="bad-http-model",
            label="Insecure Remote Model",
            provider="openai_compatible",
            base_url="http://api.groq.com/openai/v1",
            api_key_env="SOME_KEY",
            model="llama-3",
        )
    assert "must use HTTPS" in str(exc.value)

    # 2. Invalid scheme (ftp) is rejected
    with pytest.raises(ValueError) as exc:
        ModelEntry(
            id="bad-scheme-model",
            label="FTP Scheme Model",
            provider="openai_compatible",
            base_url="ftp://api.groq.com/v1",
            api_key_env="SOME_KEY",
            model="llama-3",
        )
    assert "invalid scheme" in str(exc.value)

    # 3. Missing base_url for openai_compatible is rejected
    with pytest.raises(ValueError) as exc:
        ModelEntry(
            id="missing-url-model",
            label="Missing URL Model",
            provider="openai_compatible",
            base_url=None,
            api_key_env="SOME_KEY",
            model="llama-3",
        )
    assert "base_url" in str(exc.value)

    # 4. HTTP allowed for localhost and 127.0.0.1
    local_entry = ModelEntry(
        id="local-openai",
        label="Local Ollama OpenAI Compat",
        provider="openai_compatible",
        base_url="http://localhost:11434/v1",
        api_key_env=None,
        model="llama3.2",
        local=True,
    )
    assert local_entry.base_url == "http://localhost:11434/v1"

    local_ip_entry = ModelEntry(
        id="local-ip-openai",
        label="Local IP OpenAI Compat",
        provider="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
        api_key_env=None,
        model="llama3.2",
        local=True,
    )
    assert local_ip_entry.base_url == "http://127.0.0.1:8000/v1"

    # 5. HTTPS allowed for remote
    remote_entry = ModelEntry(
        id="groq-valid",
        label="Groq",
        provider="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
        model="llama-3.3-70b",
    )
    assert remote_entry.base_url == "https://api.groq.com/openai/v1"


# ── Test 2: Key missing -> unavailable and never selected in auto mode ─────────

@pytest.mark.asyncio
async def test_key_missing_unavailable_and_never_selected_in_auto_mode(monkeypatch, clean_rate_gate):
    """If api_key_env is not set, model is unavailable and never chosen in auto mode."""
    groq_model = ModelEntry(
        id="groq-llama",
        label="Groq Llama",
        provider="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="TEST_GROQ_KEY_ENV",
        model="llama3",
        priority=1,  # Lowest number -> top priority
        rpm=30,
        rpd=1000,
    )
    gemini_model = ModelEntry(
        id="gemini-main",
        label="Gemini Main",
        provider="gemini",
        model="gemini-2.5-flash",
        priority=2,
        rpm=5,
        rpd=20,
    )

    reg = ModelRegistry([groq_model, gemini_model])
    set_model_registry(reg)

    # 1. Key is NOT set in environment
    monkeypatch.delenv("TEST_GROQ_KEY_ENV", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-key")

    assert is_model_available(groq_model) is False
    assert is_model_available(gemini_model) is True

    # Check /api/models reflects available=False and doesn't leak env var value
    client = TestClient(app)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    models_data = {m["id"]: m for m in resp.json()}
    assert models_data["groq-llama"]["available"] is False
    assert models_data["gemini-main"]["available"] is True

    # Auto mode must NOT choose groq-llama even though priority is 1
    routed = await get_provider_for(stage=2)
    assert routed.model_id == "gemini-main"

    # 2. Key is now set in environment
    monkeypatch.setenv("TEST_GROQ_KEY_ENV", "dummy-groq-key")
    assert is_model_available(groq_model) is True

    resp = client.get("/api/models")
    models_data = {m["id"]: m for m in resp.json()}
    assert models_data["groq-llama"]["available"] is True

    # Auto mode now selects groq-llama because priority is 1
    routed_now = await get_provider_for(stage=2)
    assert routed_now.model_id == "groq-llama"


# ── Test 3: 429 not retried and moves on once ─────────────────────────────────

@pytest.mark.asyncio
async def test_429_not_retried_and_moves_on_once(monkeypatch, clean_rate_gate):
    """429 raises RateLimitError immediately (no retries) and moves to fallback once."""
    call_counts = {"primary": 0, "fallback": 0}

    def mock_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "primary" in url_str:
            call_counts["primary"] += 1
            return httpx.Response(429, json={"error": {"message": "Rate limit exceeded"}})
        if "fallback" in url_str:
            call_counts["fallback"] += 1
            return httpx.Response(200, json={
                "choices": [{"message": {"role": "assistant", "content": "Fallback success response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
            })
        return httpx.Response(404)

    transport = httpx.MockTransport(mock_handler)

    monkeypatch.setenv("TEST_KEY", "dummy-key")

    primary_entry = ModelEntry(
        id="openai-primary",
        label="Primary",
        provider="openai_compatible",
        base_url="https://primary.example.com/v1",
        api_key_env="TEST_KEY",
        model="primary-model",
        priority=1,
    )
    fallback_entry = ModelEntry(
        id="openai-fallback",
        label="Fallback",
        provider="openai_compatible",
        base_url="https://fallback.example.com/v1",
        api_key_env="TEST_KEY",
        model="fallback-model",
        priority=2,
    )

    p1 = OpenAICompatibleProvider(
        base_url=primary_entry.base_url,
        api_key_env="TEST_KEY",
        model_name="primary-model",
        model_id="openai-primary",
        transport=transport,
        rate_gate=clean_rate_gate,
    )
    p2 = OpenAICompatibleProvider(
        base_url=fallback_entry.base_url,
        api_key_env="TEST_KEY",
        model_name="fallback-model",
        model_id="openai-fallback",
        transport=transport,
        rate_gate=clean_rate_gate,
    )

    routed = RoutedProvider(
        primary_entry=primary_entry,
        fallback_entry=fallback_entry,
        stage="test",
    )
    routed._primary_instance = p1
    routed._fallback_instance = p2

    result = await routed.generate("Hello world")

    assert result == "Fallback success response"
    assert routed.was_fallback is True
    assert routed.model_id == "openai-fallback"
    # Primary must have been called exactly once (NOT retried)
    assert call_counts["primary"] == 1
    assert call_counts["fallback"] == 1


# ── Test 4: Key never present in logs, errors, or API responses ────────────────

@pytest.mark.asyncio
async def test_key_never_present_in_logs_errors_or_api_responses(monkeypatch, clean_rate_gate, caplog):
    """API key secret must never be exposed in exceptions, log records, or /api/models."""
    secret_key = "sk-SUPER-SECRET-DEV-KEY-NEVER-LEAK-THIS-VALUE-123456789"
    monkeypatch.setenv("SUPER_SECRET_KEY_ENV", secret_key)

    def mock_401(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    transport = httpx.MockTransport(mock_401)

    provider = OpenAICompatibleProvider(
        base_url="https://api.groq.com/openai/v1",
        api_key_env="SUPER_SECRET_KEY_ENV",
        model_name="llama3",
        model_id="groq-test",
        transport=transport,
        rate_gate=clean_rate_gate,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate("test prompt")

    # 1. Error message must not contain secret
    err_msg = str(exc_info.value)
    assert secret_key not in err_msg
    assert "Authentication failed" in err_msg

    # 2. Logs must not contain secret
    assert secret_key not in caplog.text

    # 3. /api/models must not contain secret
    model_entry = ModelEntry(
        id="groq-test",
        label="Groq Test",
        provider="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        api_key_env="SUPER_SECRET_KEY_ENV",
        model="llama3",
    )
    set_model_registry(ModelRegistry([model_entry]))

    client = TestClient(app)
    response = client.get("/api/models")
    assert response.status_code == 200
    resp_text = response.text
    assert secret_key not in resp_text


# ── Test 5: JSON-mode parsing with repair ──────────────────────────────────────

class DemoSchema(BaseModel):
    topic: str
    key_points: list[str]


@pytest.mark.asyncio
async def test_json_mode_parsing_with_repair(monkeypatch, clean_rate_gate):
    """When supports_json_schema=False, invalid JSON output is reflected back and repaired."""
    monkeypatch.setenv("TEST_KEY", "dummy-key")
    call_count = 0

    def mock_repair_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content.decode("utf-8"))
        messages = body.get("messages", [])

        if call_count == 1:
            # First attempt: malformed JSON missing closing brackets
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": '{"topic": "FastAPI Dependency Injection", "key_points": ["modular"',
                    }
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            })
        else:
            # Second attempt: repaired valid JSON conforming to schema
            # Verify the repair prompt includes the previous validation error
            last_user_msg = messages[-1]["content"]
            assert "failed schema validation" in last_user_msg
            return httpx.Response(200, json={
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": json.dumps({
                            "topic": "FastAPI Dependency Injection",
                            "key_points": ["modular", "reusable", "type-safe"],
                        }),
                    }
                }],
                "usage": {"prompt_tokens": 35, "completion_tokens": 20},
            })

    transport = httpx.MockTransport(mock_repair_handler)

    provider = OpenAICompatibleProvider(
        base_url="https://api.example.com/v1",
        api_key_env="TEST_KEY",
        model_name="test-model",
        model_id="test-model-json",
        supports_json_schema=False,
        transport=transport,
        rate_gate=clean_rate_gate,
    )

    result = await provider.generate_structured("Explain FastAPI DI", schema=DemoSchema)

    assert isinstance(result, DemoSchema)
    assert result.topic == "FastAPI Dependency Injection"
    assert result.key_points == ["modular", "reusable", "type-safe"]
    assert call_count == 2


# ── Test 6: Usage captured into StageMetrics with registry pricing ─────────────

@pytest.mark.asyncio
async def test_usage_captured_into_stage_metrics(monkeypatch, clean_rate_gate):
    """Tokens from OpenAI usage dict are populated into StageMetrics with per-model pricing."""
    monkeypatch.setenv("TEST_KEY", "dummy-key")

    def mock_usage_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Hello"}}],
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            },
        })

    transport = httpx.MockTransport(mock_usage_handler)

    # Register model with custom registry pricing: $1.00 per 1M in, $3.00 per 1M out
    model_entry = ModelEntry(
        id="custom-priced-model",
        label="Custom Priced",
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_KEY",
        model="custom-v1",
        price_in_per_1m=1.00,
        price_out_per_1m=3.00,
    )
    set_model_registry(ModelRegistry([model_entry]))

    provider = OpenAICompatibleProvider(
        base_url=model_entry.base_url,
        api_key_env="TEST_KEY",
        model_name="custom-v1",
        model_id="custom-priced-model",
        transport=transport,
        rate_gate=clean_rate_gate,
    )

    await provider.generate("Hi")

    metrics = StageMetrics(stage_id=2, stage_name="Intent & Learning Plan")
    metrics.fill_tokens(provider)

    assert metrics.prompt_tokens == 500
    assert metrics.completion_tokens == 200
    assert metrics.total_tokens == 700
    assert metrics.model_id == "custom-priced-model"

    # Cost calculation: (500 * 1.00 + 200 * 3.00) / 1,000,000 = (500 + 600) / 1,000,000 = 0.0011 USD
    assert metrics.cost_usd == pytest.approx(0.0011, rel=1e-5)


# ── Test 7: 5xx transient retried max 2 times counted against daily budget ─────

@pytest.mark.asyncio
async def test_5xx_transient_max_2_retries_counted_against_daily_budget(monkeypatch, clean_rate_gate):
    """5xx transient errors retry up to 2 times and count every attempt against daily budget."""
    monkeypatch.setenv("TEST_KEY", "dummy-key")
    call_count = 0

    def mock_5xx_then_200(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return httpx.Response(502, text="Bad Gateway")
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "Recovered on attempt 3"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        })

    transport = httpx.MockTransport(mock_5xx_then_200)

    model_entry = ModelEntry(
        id="test-5xx-model",
        label="5xx Test Model",
        provider="openai_compatible",
        base_url="https://api.example.com/v1",
        api_key_env="TEST_KEY",
        model="model-5xx",
        rpd=20,
    )
    set_model_registry(ModelRegistry([model_entry]))

    provider = OpenAICompatibleProvider(
        base_url=model_entry.base_url,
        api_key_env="TEST_KEY",
        model_name="model-5xx",
        model_id="test-5xx-model",
        transport=transport,
        rate_gate=clean_rate_gate,
    )

    res = await provider.generate("Test prompt")
    assert res == "Recovered on attempt 3"
    assert call_count == 3
    # All 3 attempts were counted against daily budget
    assert clean_rate_gate._get_used_today("test-5xx-model") == 3
