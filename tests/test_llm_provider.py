import json
import httpx
import pytest
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
async def test_gemini_provider_stub(monkeypatch):
    """GeminiProvider adheres to LLMProvider protocol when GEMINI_MODEL is set."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-api-key")

    provider = GeminiProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "gemini-1.5-flash"
    assert provider.api_key == "test-api-key"

    text = await provider.generate("Explain Python protocols")
    assert "[Gemini:gemini-1.5-flash]" in text


@pytest.mark.asyncio
async def test_gemini_generate_structured_raises_not_implemented(monkeypatch):
    """generate_structured must raise NotImplementedError on GeminiProvider."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")
    provider = GeminiProvider()
    with pytest.raises(NotImplementedError, match="stub: real LLM call not implemented yet"):
        await provider.generate_structured("prompt", DummySchema)


def test_gemini_missing_model_raises_clear_error(monkeypatch):
    """GeminiProvider raises clear error instructing to set GEMINI_MODEL in .env if missing."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(ValueError, match="GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."):
        GeminiProvider(model_name=None)


def test_get_llm_provider_env_selection(monkeypatch):
    """get_llm_provider selects provider based on LLM_PROVIDER env var."""
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-1.5-flash")
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
