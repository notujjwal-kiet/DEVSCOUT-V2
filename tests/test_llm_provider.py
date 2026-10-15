"""
Tests for app/llm/provider.py – LLMProvider Protocol with Ollama and Gemini stubs.
"""
import pytest

from app.llm.provider import (
    GeminiProvider,
    LLMProvider,
    OllamaProvider,
    get_llm_provider,
)


@pytest.mark.asyncio
async def test_ollama_provider_stub():
    """OllamaProvider adheres to LLMProvider protocol and generates stub response."""
    provider = OllamaProvider(model_name="llama3.2")
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "llama3.2"

    text = await provider.generate("Explain FastAPI dependency injection")
    assert "[Ollama:llama3.2]" in text


@pytest.mark.asyncio
async def test_gemini_provider_stub():
    """GeminiProvider adheres to LLMProvider protocol and generates stub response."""
    provider = GeminiProvider(model_name="gemini-1.5-pro")
    assert isinstance(provider, LLMProvider)
    assert provider.model_name == "gemini-1.5-pro"

    text = await provider.generate("Explain Python protocols")
    assert "[Gemini:gemini-1.5-pro]" in text


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
