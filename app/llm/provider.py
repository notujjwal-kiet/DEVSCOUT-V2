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

    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        think: bool | None = None,
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        self.host = (
            host
            or os.getenv("OLLAMA_HOST")
            or (cfg.ollama_host if cfg else "http://localhost:11434")
        )
        self.model_name = (
            model_name
            or os.getenv("OLLAMA_MODEL")
            or (cfg.ollama_model if cfg else "llama3.2")
        )
        if think is not None:
            self.think = think
        elif os.getenv("OLLAMA_THINK") is not None:
            self.think = os.getenv("OLLAMA_THINK", "false").lower() in ("true", "1", "yes")
        elif cfg is not None:
            self.think = cfg.ollama_think
        else:
            self.think = False

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Stub generation for local Ollama."""
        return f"[Ollama:{self.model_name}@{self.host}] Pedagogical explanation stub for: {prompt[:80]}"

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: Any
    ) -> T:
        """Stub structured generation for Ollama."""
        raise NotImplementedError("stub: real LLM call not implemented yet")


class GeminiProvider:
    """Gemini model provider stub (no network calls)."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or (cfg.gemini_api_key if cfg else None)
        model = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or (cfg.gemini_model if cfg else None)
        )
        if not model:
            raise ValueError(
                "GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."
            )
        self.model_name = model

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: Any) -> str:
        """Stub generation for Gemini."""
        return f"[Gemini:{self.model_name}] Pedagogical explanation stub for: {prompt[:80]}"

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: Any
    ) -> T:
        """Stub structured generation for Gemini."""
        raise NotImplementedError("stub: real LLM call not implemented yet")


def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """
    Factory function returning the configured LLMProvider.
    Selected via provider_name or LLM_PROVIDER env var ('ollama' | 'gemini').
    """
    cfg = None
    try:
        from app.config import get_settings

        cfg = get_settings()
    except Exception:
        pass

    chosen = (
        provider_name
        or os.getenv("LLM_PROVIDER")
        or (cfg.llm_provider if cfg else "ollama")
    ).lower().strip()

    if chosen == "gemini":
        return GeminiProvider(model_name=model_name, **kwargs)
    elif chosen == "ollama":
        return OllamaProvider(model_name=model_name, **kwargs)
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{chosen}'. Supported options are 'ollama' and 'gemini'."
        )
