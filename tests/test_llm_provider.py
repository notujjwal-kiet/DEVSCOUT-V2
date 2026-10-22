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
    OpenAICompatibleProvider,
    get_llm_provider,
)


class DummySchema(BaseModel):
    message: str
    confidence: float





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
    monkeypatch.setattr("app.config.get_settings", lambda: MagicMock(gemini_model=None, gemini_api_key="test-key"))
    with pytest.raises(LLMProviderError, match="GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."):
        GeminiProvider(model_name=None, api_key="test-key")


def test_gemini_missing_api_key_raises_clear_error(monkeypatch):
    """GeminiProvider raises clear LLMProviderError if GEMINI_API_KEY is missing."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("app.config.get_settings", lambda: MagicMock(gemini_model="gemini-test", gemini_api_key=None))
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

    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    provider = get_llm_provider(base_url="https://api.openai.com/v1", model_name="gpt-4o")
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_name == "gpt-4o"


def test_get_llm_provider_invalid():
    """get_llm_provider raises ValueError for unsupported provider."""
    with pytest.raises(ValueError):
        get_llm_provider(provider_name="unsupported-provider")


def test_get_llm_provider_accepts_rate_gate_and_kwargs(monkeypatch):
    """get_llm_provider forwards rate_gate and unknown kwargs without TypeError."""
    from unittest.mock import MagicMock

    mock_gate = MagicMock()
    p_gemini = get_llm_provider(
        provider_name="gemini",
        model_name="gemini-2.5-flash",
        api_key="fake-key",
        rate_gate=mock_gate,
        extra_param="extra_value",
    )
    assert isinstance(p_gemini, GeminiProvider)
    assert p_gemini.rate_gate is mock_gate

    p_openai = get_llm_provider(
        provider_name="openai_compatible",
        model_name="qwen",
        base_url="https://api.example.com/v1",
        rate_gate=mock_gate,
        extra_param="extra_value",
    )
    assert isinstance(p_openai, OpenAICompatibleProvider)
    assert p_openai.rate_gate is mock_gate

