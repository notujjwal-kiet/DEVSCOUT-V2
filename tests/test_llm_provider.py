import json
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from google.genai import errors
from pydantic import BaseModel, ValidationError

from app.llm.provider import (
    GeminiProvider,
    LLMProvider,
    LLMProviderError,
    OllamaProvider,
    get_llm_provider,
)


class DummySchema(BaseModel):
    message: str
    confidence: float


def test_ollama_provider_config(monkeypatch):
    """OllamaProvider adheres to LLMProvider protocol and reads host/model/think from config."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen-test")
    monkeypatch.setenv("OLLAMA_THINK", "true")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "45")

    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "qwen-test"
    assert provider.host == "http://127.0.0.1:11434"
    assert provider.think is True
    assert provider.timeout == 45.0


@pytest.mark.asyncio
async def test_ollama_generate_success():
    """generate sends model, messages, think, and returns assistant message content as str."""
    recorded_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert body["think"] is True
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "You are a helpful tutor."
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "Explain token bucket algorithm"
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "Token bucket controls traffic by refilling tokens at a fixed rate.",
                },
                "done": True,
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        host="http://mock-ollama:11434",
        model_name="test-model",
        think=True,
        transport=transport,
    )
    result = await provider.generate(
        prompt="Explain token bucket algorithm",
        system_prompt="You are a helpful tutor.",
    )
    assert result == "Token bucket controls traffic by refilling tokens at a fixed rate."
    assert len(recorded_requests) == 1
    assert recorded_requests[0].url == "http://mock-ollama:11434/api/chat"


@pytest.mark.asyncio
async def test_ollama_generate_structured_success():
    """generate_structured sends schema format, parses JSON, and returns validated schema."""
    recorded_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "test-model"
        assert body["stream"] is False
        assert "format" in body
        assert body["format"] == DummySchema.model_json_schema()
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": '{"message": "Structured explanation", "confidence": 0.95}',
                },
                "done": True,
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        host="http://mock-ollama:11434",
        model_name="test-model",
        transport=transport,
    )
    result = await provider.generate_structured(
        prompt="Provide structured summary",
        schema=DummySchema,
    )
    assert isinstance(result, DummySchema)
    assert result.message == "Structured explanation"
    assert result.confidence == 0.95
    assert len(recorded_requests) == 1


@pytest.mark.asyncio
async def test_ollama_connection_error():
    """On connection errors, OllamaProvider raises clear LLMProviderError mentioning host."""
    failed_host = "http://unreachable-ollama:11434"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by peer")

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        host=failed_host,
        model_name="test-model",
        transport=transport,
    )

    with pytest.raises(LLMProviderError) as exc_info:
        await provider.generate(prompt="Test prompt")

    err_msg = str(exc_info.value)
    assert failed_host in err_msg
    assert "Failed to connect to Ollama host" in err_msg


@pytest.mark.asyncio
async def test_ollama_generate_structured_invalid_json():
    """If Ollama returns invalid JSON content, schema.model_validate_json raises ValidationError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "test-model",
                "message": {
                    "role": "assistant",
                    "content": "This is raw text, not valid JSON for the schema",
                },
                "done": True,
            },
        )

    transport = httpx.MockTransport(handler)
    provider = OllamaProvider(
        host="http://mock-ollama:11434",
        model_name="test-model",
        transport=transport,
    )

    with pytest.raises(ValidationError):
        await provider.generate_structured(
            prompt="Structured prompt",
            schema=DummySchema,
        )


@pytest.mark.asyncio
async def test_gemini_generate_success():
    """generate requests plain text from Gemini client and returns text."""
    mock_client = MagicMock()
    mock_resp = MagicMock(text="Gemini plain text explanation.")
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    result = await provider.generate(
        prompt="Explain rate limiting",
        system_prompt="Be concise",
    )
    assert result == "Gemini plain text explanation."
    mock_client.aio.models.generate_content.assert_awaited_once()
    kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"] == "Explain rate limiting"
    assert kwargs["config"].system_instruction == "Be concise"


@pytest.mark.asyncio
async def test_gemini_generate_structured_success():
    """generate_structured requests JSON output with schema and validates with model_validate_json."""
    mock_client = MagicMock()
    mock_resp = MagicMock(text='{"message": "Valid structured JSON", "confidence": 0.98}')
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    result = await provider.generate_structured(
        prompt="Generate structured summary",
        schema=DummySchema,
        system_prompt="Return JSON",
    )
    assert isinstance(result, DummySchema)
    assert result.message == "Valid structured JSON"
    assert result.confidence == 0.98

    kwargs = mock_client.aio.models.generate_content.await_args.kwargs
    assert kwargs["model"] == "gemini-test"
    assert kwargs["contents"] == "Generate structured summary"
    assert kwargs["config"].response_mime_type == "application/json"
    assert kwargs["config"].response_schema == DummySchema
    assert kwargs["config"].system_instruction == "Return JSON"


@pytest.mark.asyncio
async def test_gemini_generate_structured_invalid_json():
    """If Gemini returns invalid JSON content, schema.model_validate_json raises ValidationError."""
    mock_client = MagicMock()
    mock_resp = MagicMock(text="Non-JSON raw text")
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_resp)

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    with pytest.raises(ValidationError):
        await provider.generate_structured(
            prompt="Generate structured summary",
            schema=DummySchema,
        )


def test_gemini_missing_model_raises_clear_error(monkeypatch):
    """GeminiProvider raises clear LLMProviderError if GEMINI_MODEL is missing."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(LLMProviderError, match="GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."):
        GeminiProvider(model_name=None, api_key="test-key")


def test_gemini_missing_api_key_raises_clear_error(monkeypatch):
    """GeminiProvider raises clear LLMProviderError if GEMINI_API_KEY is missing."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMProviderError, match="GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in .env."):
        GeminiProvider(model_name="gemini-test", api_key=None)


@pytest.mark.asyncio
async def test_gemini_rate_limit_error_mapping():
    """SDK 429 rate limit errors map to LLMProviderError with a readable message."""
    mock_client = MagicMock()
    api_err = errors.APIError(429, {"error": {"message": "Quota depleted", "status": "RESOURCE_EXHAUSTED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=api_err)

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    with pytest.raises(LLMProviderError, match="rate limit exceeded"):
        await provider.generate("prompt")


@pytest.mark.asyncio
async def test_gemini_auth_error_mapping():
    """SDK 401/403 auth errors map to LLMProviderError with a readable message."""
    mock_client = MagicMock()
    api_err = errors.APIError(403, {"error": {"message": "Invalid API key", "status": "PERMISSION_DENIED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=api_err)

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    with pytest.raises(LLMProviderError, match="authentication failed"):
        await provider.generate("prompt")


@pytest.mark.asyncio
async def test_gemini_network_error_mapping():
    """SDK network/connection errors map to LLMProviderError with a readable message."""
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=httpx.ConnectError("Failed to connect to Gemini endpoint")
    )

    provider = GeminiProvider(
        model_name="gemini-test",
        api_key="test-api-key",
        client=mock_client,
    )
    with pytest.raises(LLMProviderError, match="network error"):
        await provider.generate("prompt")


def test_get_llm_provider_env_selection(monkeypatch):
    """get_llm_provider selects provider based on LLM_PROVIDER env var."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")
    provider = get_llm_provider()
    assert isinstance(provider, GeminiProvider)
    assert provider.model_name == "gemini-1.5-flash"

    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "mistral-small")
    provider = get_llm_provider()
    assert isinstance(provider, OllamaProvider)
    assert provider.model_name == "mistral-small"


def test_get_llm_provider_invalid():
    """get_llm_provider raises ValueError for unsupported provider."""
    with pytest.raises(ValueError):
        get_llm_provider(provider_name="unsupported-provider")
