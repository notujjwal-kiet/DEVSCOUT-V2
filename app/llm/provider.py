"""
app/llm/provider.py

LLM Provider interface and stubs for DEVSCOUT.
Protocol LLMProvider with Ollama and Gemini stub implementations (no network calls).
Provider is chosen by the LLM_PROVIDER environment variable; model names are configurable.
"""
from __future__ import annotations

import os
from typing import Protocol, TypeVar, runtime_checkable

from google import genai
from google.genai import errors, types
import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(ValueError):
    """Raised when an LLM provider encounters a network, connection, configuration, or execution error."""


class RateLimitError(LLMProviderError):
    """Raised when an LLM provider encounters a 429 rate limit or quota exhaustion error."""


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
            if exc.response.status_code == 429 or "quota" in exc.response.text.lower():
                raise RateLimitError(
                    f"Ollama host '{self.host}' rate limit exceeded (HTTP 429): {exc.response.text}"
                ) from exc
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


def _map_gemini_error(exc: Exception, model_name: str) -> LLMProviderError:
    """Map Gemini SDK and HTTP/transport errors to a readable LLMProviderError."""
    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None) or str(exc)
        if code == 429 or "RESOURCE_EXHAUSTED" in str(exc) or "quota" in str(exc).lower():
            return RateLimitError(
                f"Gemini API rate limit exceeded for model '{model_name}': {message}"
            )
        if (
            code in (401, 403)
            or "PERMISSION_DENIED" in str(exc)
            or "UNAUTHENTICATED" in str(exc)
            or "API_KEY_INVALID" in str(exc)
        ):
            return LLMProviderError(
                f"Gemini API authentication failed for model '{model_name}': {message}"
            )
        return LLMProviderError(
            f"Gemini API error ({code}) for model '{model_name}': {message}"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429 or "quota" in exc.response.text.lower():
            return RateLimitError(
                f"Gemini API rate limit exceeded (HTTP 429) for model '{model_name}': {exc.response.text}"
            )
        return LLMProviderError(
            f"Gemini HTTP error ({exc.response.status_code}) for model '{model_name}': {exc}"
        )
    if isinstance(exc, httpx.RequestError):
        return LLMProviderError(
            f"Gemini network error for model '{model_name}': {exc}"
        )
    exc_lower = str(exc).lower()
    if "429" in exc_lower or "quota" in exc_lower or "resource_exhausted" in exc_lower:
        return RateLimitError(
            f"Gemini API rate limit exceeded for model '{model_name}': {exc}"
        )
    return LLMProviderError(
        f"Gemini request failed for model '{model_name}': {exc}"
    )


class GeminiProvider:
    """Gemini model provider using official google-genai Python SDK."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        client: genai.Client | None = None,
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        model = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or (cfg.gemini_model if cfg else None)
        )
        if not model:
            raise LLMProviderError(
                "GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."
            )
        self.model_name = model

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or (cfg.gemini_api_key if cfg else None)
        )
        if not self.api_key and client is None:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in .env."
            )

        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        else:
            self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in .env."
            )
        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def get_client(self) -> genai.Client:
        """Return the underlying genai.Client instance."""
        return self._get_client()

    async def generate_with_search(
        self, prompt: str, **kwargs: object
    ) -> types.GenerateContentResponse:
        """
        Generate content using Gemini with Google Search tool enabled.
        Plain text call only; does not use response_schema or JSON mode.
        """
        client = self._get_client()
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )
        try:
            return await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            raise _map_gemini_error(exc, self.model_name) from exc


    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> str:
        """Generate plain text from prompt using Gemini model."""
        client = self._get_client()
        config = (
            types.GenerateContentConfig(system_instruction=system_prompt)
            if system_prompt
            else None
        )
        try:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return response.text or ""
        except Exception as exc:
            raise _map_gemini_error(exc, self.model_name) from exc

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> T:
        """Generate structured output adhering to schema using Gemini model."""
        client = self._get_client()
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
        )
        try:
            response = await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            text = response.text or ""
        except Exception as exc:
            raise _map_gemini_error(exc, self.model_name) from exc

        return schema.model_validate_json(text)


def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    **kwargs: object,
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
