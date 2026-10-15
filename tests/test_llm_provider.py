"""
Tests for app/llm/provider.py – LLMProvider Protocol with Ollama and Gemini stubs.
"""
import pytest
from pydantic import BaseModel

from app.llm.provider import (
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    get_llm_provider,
)


class DummySchema(BaseModel):
    message: str


@pytest.mark.asyncio
async def test_ollama_provider_stub(monkeypatch):
    """OllamaProvider adheres to LLMProvider protocol and reads host/model/think from config."""
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setenv("OLLAMA_THINK", "true")

    provider = OllamaProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "qwen3.5:4b"
    assert provider.host == "http://127.0.0.1:11434"
    assert provider.think is True

    text = await provider.generate("Explain FastAPI dependency injection")
    assert "[Ollama:qwen3.5:4b@http://127.0.0.1:11434]" in text


@pytest.mark.asyncio
async def test_ollama_generate_structured_raises_not_implemented():
    """generate_structured must raise NotImplementedError instead of confusing schema errors."""
    provider = OllamaProvider(model_name="llama3.2")
    with pytest.raises(NotImplementedError, match="stub: real LLM call not implemented yet"):
        await provider.generate_structured("prompt", DummySchema)


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
