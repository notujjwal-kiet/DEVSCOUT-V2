"""
app/llm/provider.py

LLM Provider interface and stubs for DEVSCOUT.
Protocol LLMProvider with Ollama and Gemini stub implementations (no network calls).
Provider is chosen by the LLM_PROVIDER environment variable; model names are configurable.
"""
from __future__ import annotations

import os
from typing import Any, Protocol, TypeVar, runtime_checkable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the LLM provider interface."""

    model_name: str

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Generate plain text from prompt."""
        ...

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: Any
    ) -> T:
        """Generate structured output adhering to a Pydantic schema."""
        ...


class OllamaProvider:
    """Ollama local model provider stub (no network calls)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Stub generation for local Ollama."""
        return f"[Ollama:{self.model_name}] Pedagogical explanation stub for: {prompt[:80]}"

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: Any
    ) -> T:
        """Stub structured generation for Ollama."""
        # Instantiate schema with defaults or minimal valid data
        return schema.model_validate({}) if hasattr(schema, "model_validate") else schema()


class GeminiProvider:
    """Gemini model provider stub (no network calls)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Stub generation for Gemini."""
        return f"[Gemini:{self.model_name}] Pedagogical explanation stub for: {prompt[:80]}"

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: Any
    ) -> T:
        """Stub structured generation for Gemini."""
        return schema.model_validate({}) if hasattr(schema, "model_validate") else schema()


def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> LLMProvider:
    """
    Factory function returning the configured LLMProvider.
    Selected via provider_name or LLM_PROVIDER env var ('ollama' | 'gemini').
    """
    chosen = (provider_name or os.getenv("LLM_PROVIDER", "ollama")).lower().strip()

    if chosen == "gemini":
        return GeminiProvider(model_name=model_name)
    elif chosen == "ollama":
        return OllamaProvider(model_name=model_name)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{chosen}'. Supported options are 'ollama' and 'gemini'."
        )
