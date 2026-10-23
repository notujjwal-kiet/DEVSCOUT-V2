"""
app/providers/openai_compatible.py

OpenAI-compatible HTTP provider using httpx.AsyncClient to call
configurable /chat/completions endpoints (Groq, Cerebras, Ollama, etc.).

API keys and secrets are never logged.
"""
from __future__ import annotations

import re
from typing import Any
import httpx
import structlog

from app.providers.base import (
    AuthenticationError,
    BaseChatProvider,
    ChatMessage,
    ChatResponse,
    ConfigurationError,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = structlog.get_logger(__name__)


def _parse_retry_after(header_val: str | None) -> float | None:
    """Parse Retry-After header value into seconds."""
    if not header_val:
        return None
    header_clean = header_val.strip()
    try:
        # Standard seconds integer/float
        val = float(header_clean)
        return max(0.0, val)
    except ValueError:
        pass
    # If HTTP-date, fallback to a safe 30s backoff
    return 30.0


class OpenAICompatibleProvider(BaseChatProvider):
    """
    Generic provider for services implementing the OpenAI Chat Completions API specification:
    - Groq (https://api.groq.com/openai/v1)
    - Cerebras (https://api.cerebras.ai/v1)
    - Ollama (http://localhost:11434/v1)
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tier: str = "primary",
        is_local: bool = False,
        price_in_per_1m: float | None = None,
        price_out_per_1m: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            tier=tier,
            is_local=is_local,
            timeout_seconds=timeout_seconds,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m

    def is_available(self) -> bool:
        """Check if provider is configured and available for requests."""
        if self.is_local:
            return bool(self.base_url)
        return bool(self.api_key and self.base_url)

    def get_info(self) -> dict[str, Any]:
        """Return safe public metadata without secrets."""
        return {
            "id": f"{self.name}-{self.model}",
            "label": f"{self.name.capitalize()} ({self.model})",
            "provider": self.name,
            "local": self.is_local,
            "tier": self.tier,
            "available": self.is_available(),
            "used_today": 0,
            "limit_today": None,
            "price_in_per_1m": self.price_in_per_1m,
            "price_out_per_1m": self.price_out_per_1m,
        }

    def _get_endpoint(self) -> str:
        """Compute the full chat completions endpoint URL."""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def chat(
        self,
        messages: list[ChatMessage],
        task: str = "chat",
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute chat completion via HTTP POST with bounded timeout and safe error handling."""
        if not self.is_available():
            raise ConfigurationError(
                f"Provider '{self.name}' is not configured or missing API credentials.",
                provider=self.name,
                model=self.model,
            )

        endpoint = self._get_endpoint()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens or "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        # Allow passing custom provider options if supported
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        logger.debug(
            "openai_compatible_chat_request",
            provider=self.name,
            model=self.model,
            endpoint=endpoint,
            message_count=len(messages),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning(
                "openai_compatible_timeout",
                provider=self.name,
                model=self.model,
                timeout=self.timeout_seconds,
            )
            raise ProviderTimeoutError(
                f"Request to {self.name} timed out after {self.timeout_seconds}s.",
                provider=self.name,
                model=self.model,
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "openai_compatible_network_error",
                provider=self.name,
                model=self.model,
                error=str(exc),
            )
            raise ProviderUnavailableError(
                f"Network connection to {self.name} failed: {exc}",
                provider=self.name,
                model=self.model,
            ) from exc

        status_code = response.status_code

        # ── Error Mapping ─────────────────────────────────────────────
        if status_code in (401, 403):
            # Authentication failure: Non-retryable against the same provider
            logger.error(
                "openai_compatible_auth_error",
                provider=self.name,
                model=self.model,
                status=status_code,
            )
            raise AuthenticationError(
                f"Authentication failed for {self.name} (HTTP {status_code}). Check API credentials.",
                provider=self.name,
                model=self.model,
            )

        if status_code == 429:
            # Rate limit or quota exhaustion: Eligible for fallback, respect Retry-After
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            logger.warning(
                "openai_compatible_rate_limit",
                provider=self.name,
                model=self.model,
                retry_after=retry_after,
            )
            raise RateLimitError(
                f"Rate limit exceeded for {self.name} (HTTP 429).",
                provider=self.name,
                model=self.model,
                retry_after=retry_after,
            )

        if status_code in (400, 422):
            # Malformed request or invalid parameters: Non-retryable on same provider
            err_msg = response.text[:200]
            logger.warning(
                "openai_compatible_invalid_request",
                provider=self.name,
                model=self.model,
                status=status_code,
                detail=err_msg,
            )
            raise InvalidRequestError(
                f"Invalid request to {self.name} (HTTP {status_code}): {err_msg}",
                provider=self.name,
                model=self.model,
            )

        if status_code >= 500:
            # Server outage: Eligible for fallback
            logger.warning(
                "openai_compatible_server_error",
                provider=self.name,
                model=self.model,
                status=status_code,
            )
            raise ProviderUnavailableError(
                f"{self.name} service is temporarily unavailable (HTTP {status_code}).",
                provider=self.name,
                model=self.model,
            )

        if status_code != 200:
            raise ProviderUnavailableError(
                f"Unexpected status from {self.name} (HTTP {status_code}).",
                provider=self.name,
                model=self.model,
            )

        # ── Success Parsing ───────────────────────────────────────────
        try:
            data = response.json()
            choice = data["choices"][0]
            message_obj = choice.get("message", {})
            content = message_obj.get("content", "")
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage")
        except (KeyError, IndexError, ValueError) as exc:
            logger.error(
                "openai_compatible_malformed_response",
                provider=self.name,
                model=self.model,
                error=str(exc),
            )
            raise ProviderUnavailableError(
                f"Failed to parse response from {self.name}.",
                provider=self.name,
                model=self.model,
            ) from exc

        return ChatResponse(
            content=content,
            provider=self.name,
            model=self.model,
            task=task,
            research_backed=False,  # OpenAI-compatible chat endpoints do not claim research without verification
            finish_reason=finish_reason,
            usage=usage,
        )
