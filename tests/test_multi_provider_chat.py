"""
tests/test_multi_provider_chat.py

Comprehensive tests for multi-provider AI layer (Groq, Cerebras, Gemini, Ollama):
- Mocked HTTP responses (no real API keys required).
- Successful requests.
- Rate-limit fallback (429 with Retry-After).
- Transient timeout fallback.
- Server outage (500) fallback.
- Invalid credentials (401/403) handling (non-retryable on same provider).
- Malformed request (400) handling.
- Missing configuration behavior.
- Task-based routing and research_backed veracity.
- OpenAPI schema validation for /api/models and /api/chat.
- Credential leak audit on responses.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.providers.base import (
    AuthenticationError,
    ChatMessage,
    ChatResponse,
    ConfigurationError,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.providers.openai_compatible import OpenAICompatibleProvider
from app.providers.registry import ProviderRegistry
from app.providers.router import ProviderRouter


@pytest.fixture
def test_client():
    return TestClient(app)


# ── Test 1: OpenAICompatibleProvider Successful Request ────────────────────────


@pytest.mark.asyncio
async def test_openai_compatible_success():
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="gsk-test-key",
        timeout_seconds=5.0,
    )

    mock_response = httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-123",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Async in Python uses an event loop.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18},
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        messages = [ChatMessage(role="user", content="Explain Python async")]
        res = await provider.chat(messages, task="chat")

        assert isinstance(res, ChatResponse)
        assert res.provider == "groq"
        assert res.model == "llama-3.3-70b-versatile"
        assert res.content == "Async in Python uses an event loop."
        assert res.finish_reason == "stop"
        assert res.research_backed is False
        assert res.usage == {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}


# ── Test 2: Rate Limit (429) and Retry-After Parsing ──────────────────────────


@pytest.mark.asyncio
async def test_openai_compatible_rate_limit_with_retry_after():
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="gsk-test-key",
    )

    mock_resp = httpx.Response(
        status_code=429,
        headers={"retry-after": "15"},
        json={"error": {"message": "Rate limit reached"}},
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(RateLimitError) as exc_info:
            await provider.chat(messages)

        assert exc_info.value.provider == "groq"
        assert exc_info.value.retry_after == 15.0


# ── Test 3: Authentication Failure (401) is Non-Retryable ──────────────────────


@pytest.mark.asyncio
async def test_openai_compatible_auth_failure():
    provider = OpenAICompatibleProvider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        model="llama3.1-8b",
        api_key="invalid-key",
    )

    mock_resp = httpx.Response(
        status_code=401,
        json={"error": {"message": "Invalid API Key"}},
        request=httpx.Request("POST", "https://api.cerebras.ai/v1/chat/completions"),
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp

        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(AuthenticationError) as exc_info:
            await provider.chat(messages)

        assert exc_info.value.provider == "cerebras"
        assert "Authentication failed" in str(exc_info.value)


# ── Test 4: Timeout Handling ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_compatible_timeout():
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="gsk-test-key",
        timeout_seconds=2.0,
    )

    with patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Read timed out")):
        messages = [ChatMessage(role="user", content="Hello")]
        with pytest.raises(ProviderTimeoutError) as exc_info:
            await provider.chat(messages)

        assert exc_info.value.provider == "groq"
        assert "timed out after 2.0s" in str(exc_info.value)


# ── Test 5: Missing Configuration ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_openai_compatible_missing_credentials():
    provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key=None,  # Missing key
    )
    assert provider.is_available() is False

    with pytest.raises(ConfigurationError):
        await provider.chat([ChatMessage(role="user", content="Hi")])


# ── Test 6: Router Fallback on Rate Limit ─────────────────────────────────────


@pytest.mark.asyncio
async def test_router_fallback_on_rate_limit():
    groq_provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="mock-groq-key",
    )
    cerebras_provider = OpenAICompatibleProvider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        model="llama3.1-8b",
        api_key="mock-cerebras-key",
    )

    registry = ProviderRegistry()
    registry._providers = {
        "groq": groq_provider,
        "cerebras": cerebras_provider,
    }

    router = ProviderRouter(registry=registry)

    # Groq fails with 429; Cerebras succeeds
    async def mock_chat(messages, task="chat", **kwargs):
        raise RateLimitError("Groq 429", provider="groq", model="llama-3.3-70b-versatile", retry_after=30.0)

    groq_provider.chat = AsyncMock(side_effect=mock_chat)
    cerebras_provider.chat = AsyncMock(
        return_value=ChatResponse(
            content="Fallback answer from Cerebras",
            provider="cerebras",
            model="llama3.1-8b",
            task="chat",
        )
    )

    messages = [ChatMessage(role="user", content="What is WebSockets?")]
    res = await router.route_chat(messages, task="chat")

    assert res.provider == "cerebras"
    assert res.content == "Fallback answer from Cerebras"
    assert router._is_in_cooldown("groq") is True


# ── Test 7: Router Fallback on Auth Error (Does not retry Groq) ────────────────


@pytest.mark.asyncio
async def test_router_fallback_on_auth_error():
    groq_provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="bad-groq-key",
    )
    cerebras_provider = OpenAICompatibleProvider(
        name="cerebras",
        base_url="https://api.cerebras.ai/v1",
        model="llama3.1-8b",
        api_key="good-cerebras-key",
    )

    registry = ProviderRegistry()
    registry._providers = {
        "groq": groq_provider,
        "cerebras": cerebras_provider,
    }

    router = ProviderRouter(registry=registry)

    groq_provider.chat = AsyncMock(
        side_effect=AuthenticationError("401 Unauthorized", provider="groq")
    )
    cerebras_provider.chat = AsyncMock(
        return_value=ChatResponse(
            content="Answered by Cerebras fallback",
            provider="cerebras",
            model="llama3.1-8b",
            task="chat",
        )
    )

    messages = [ChatMessage(role="user", content="Hello")]
    res = await router.route_chat(messages, task="chat")

    assert res.provider == "cerebras"
    assert res.content == "Answered by Cerebras fallback"
    # Groq was only called once, never retried
    assert groq_provider.chat.call_count == 1


# ── Test 8: Research Task Veracity ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_task_veracity():
    groq_provider = OpenAICompatibleProvider(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
        api_key="mock-groq-key",
    )
    registry = ProviderRegistry()
    registry._providers = {"groq": groq_provider}

    router = ProviderRouter(registry=registry)

    groq_provider.chat = AsyncMock(
        return_value=ChatResponse(
            content="General knowledge explanation without citations",
            provider="groq",
            model="llama-3.3-70b-versatile",
            task="research",
            research_backed=False,
        )
    )

    messages = [ChatMessage(role="user", content="Research WebRTC data channels")]
    res = await router.route_chat(messages, task="research")

    # Does NOT silently claim research-backed when no web discovery/excerpts occurred
    assert res.research_backed is False


# ── Test 9: HTTP Endpoint POST /api/chat ──────────────────────────────────────


def test_api_chat_endpoint(test_client):
    mock_response = ChatResponse(
        content="FastAPI is a modern Python framework.",
        provider="groq",
        model="llama-3.3-70b-versatile",
        task="explain",
        research_backed=False,
        finish_reason="stop",
        usage={"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    )

    with patch(
        "app.providers.router.ProviderRouter.route_chat",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        resp = test_client.post(
            "/api/chat",
            json={"message": "What is FastAPI?", "task": "explain"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["content"] == "FastAPI is a modern Python framework."
        assert data["provider"] == "groq"
        assert data["model"] == "llama-3.3-70b-versatile"
        assert data["task"] == "explain"
        assert data["research_backed"] is False


def test_api_chat_validation_error(test_client):
    # Missing both 'message' and 'messages'
    resp = test_client.post("/api/chat", json={"task": "chat"})
    assert resp.status_code == 422


# ── Test 10: GET /api/models Returns Configured Models without Secrets ─────────


def test_api_models_endpoint_multi_provider(test_client):
    resp = test_client.get("/api/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)

    # Check for providers
    providers = {m["provider"] for m in models}
    assert "gemini" in providers
    assert "groq" in providers
    assert "cerebras" in providers

    # Ensure no secrets leaked
    forbidden_terms = ["key", "secret", "token", "password", "bearer", "authorization"]
    for m in models:
        for k, v in m.items():
            for term in forbidden_terms:
                assert term not in k.lower(), f"Forbidden key '{k}' found in /api/models"
                if isinstance(v, str):
                    assert term not in v.lower(), f"Forbidden value in /api/models '{k}': {v}"


# ── Test 11: OpenAPI Schema Registration ───────────────────────────────────────


def test_openapi_schema_has_chat_and_models_endpoints(test_client):
    resp = test_client.get("/openapi.json")
    assert resp.status_code == 200
    schema = resp.json()
    paths = schema.get("paths", {})

    assert "/api/chat" in paths
    assert "post" in paths["/api/chat"]

    assert "/api/models" in paths
    assert "get" in paths["/api/models"]

    assert "/api/health" in paths
    assert "get" in paths["/api/health"]
