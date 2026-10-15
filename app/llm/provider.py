"""
app/llm/provider.py

LLM Provider interface and stubs for DEVSCOUT.
Protocol LLMProvider with Ollama and Gemini stub implementations (no network calls).
Provider is chosen by the LLM_PROVIDER environment variable; model names are configurable.
"""
from __future__ import annotations

import os
from typing import Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters a network, connection, or execution error."""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the LLM provider interface."""

    model_name: str

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
        """Generate plain text from prompt."""
        ...

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: object
    ) -> T:
        """Generate structured output adhering to a Pydantic schema."""
        ...


class OllamaProvider:
    """Ollama local model provider using httpx async HTTP API."""

    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        think: bool | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
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
        elif cfg is not None and hasattr(cfg, "ollama_think"):
            self.think = cfg.ollama_think
        else:
            self.think = False

        if timeout is not None:
            self.timeout = float(timeout)
        elif os.getenv("OLLAMA_TIMEOUT") is not None:
            self.timeout = float(os.getenv("OLLAMA_TIMEOUT"))
        elif cfg is not None and hasattr(cfg, "ollama_timeout"):
            self.timeout = float(cfg.ollama_timeout)
        else:
            self.timeout = 120.0

        if client is not None:
            self._client = client
        elif transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=self.timeout)
        else:
            self._client = None

    async def _post_chat(self, payload: dict[str, object]) -> httpx.Response:
        url = f"{self.host.rstrip('/')}/api/chat"
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
            response.raise_for_status()
            return response
        except httpx.RequestError as exc:
            raise LLMProviderError(
                f"Failed to connect to Ollama host at '{self.host}': {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise LLMProviderError(
                f"Ollama host '{self.host}' returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> str:
        """Generate plain text from prompt using Ollama chat endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,
        }
        response = await self._post_chat(payload)
        try:
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to parse JSON response from Ollama host at '{self.host}': {exc}"
            ) from exc

        message = data.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(data.get("response", ""))

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> T:
        """Generate structured output adhering to schema using Ollama chat endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "format": schema.model_json_schema(),
        }
        response = await self._post_chat(payload)
        try:
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to parse JSON response from Ollama host at '{self.host}': {exc}"
            ) from exc

        message = data.get("message")
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        else:
            content = str(data.get("response", ""))

        return schema.model_validate_json(content)


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
