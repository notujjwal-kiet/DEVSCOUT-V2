"""
app/providers/gemini_provider.py

Google Gemini as a separately configured research and chat provider.
Integrates with google-genai SDK or direct HTTP API with bounded timeouts and error mapping.
Never claims research_backed=True unless research or evidence discovery was actually performed.
"""
from __future__ import annotations

import asyncio
from typing import Any
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


class GeminiChatProvider(BaseChatProvider):
    """
    Separately configured research and chat provider using Google Gemini.
    """

    def __init__(
        self,
        name: str = "gemini",
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        tier: str = "research",
        rpd_limit: int | None = 20,
        rpm_limit: int | None = 5,
        price_in_per_1m: float | None = 0.075,
        price_out_per_1m: float | None = 0.30,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            tier=tier,
            is_local=False,
            timeout_seconds=timeout_seconds,
        )
        self.api_key = api_key
        self.rpd_limit = rpd_limit
        self.rpm_limit = rpm_limit
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m

    def is_available(self) -> bool:
        """Check if Gemini API key is configured."""
        return bool(self.api_key and self.api_key.strip())

    def get_info(self) -> dict[str, Any]:
        """Return safe public metadata without secrets."""
        return {
            "id": f"{self.name}-{self.model}",
            "label": f"Google Gemini ({self.model})",
            "provider": self.name,
            "local": self.is_local,
            "tier": self.tier,
            "available": self.is_available(),
            "used_today": 0,
            "limit_today": self.rpd_limit,
            "price_in_per_1m": self.price_in_per_1m,
            "price_out_per_1m": self.price_out_per_1m,
        }

    async def chat(
        self,
        messages: list[ChatMessage],
        task: str = "chat",
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Execute chat or research generation via Gemini.
        If task == 'research', evidence grounding flag is set if research was supplied/conducted.
        """
        if not self.is_available():
            raise ConfigurationError(
                "Gemini provider is not configured. GEMINI_API_KEY is missing.",
                provider=self.name,
                model=self.model,
            )

        # Build prompt from messages
        prompt_parts: list[str] = []
        for m in messages:
            if m.role == "system":
                prompt_parts.append(f"System: {m.content}\n")
            elif m.role == "assistant":
                prompt_parts.append(f"Assistant: {m.content}\n")
            else:
                prompt_parts.append(f"User: {m.content}\n")
        prompt = "\n".join(prompt_parts)

        # Handle research context if supplied
        research_performed = kwargs.get("research_performed", False)
        research_sources = kwargs.get("research_sources", [])
        if research_sources:
            research_performed = True

        logger.debug(
            "gemini_chat_request",
            model=self.model,
            task=task,
            message_count=len(messages),
            research_performed=research_performed,
        )

        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)

            # Enforce bounded timeout
            response = await asyncio.wait_for(
                client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=kwargs.get("temperature", 0.7),
                        max_output_tokens=kwargs.get("max_tokens", None),
                    ),
                ),
                timeout=self.timeout_seconds,
            )
            content = response.text or ""
            usage_dict = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage_dict = {
                    "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                    "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                    "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
                }

            return ChatResponse(
                content=content,
                provider=self.name,
                model=self.model,
                task=task,
                # Crucial requirement: Do not silently claim research-backed unless research occurred
                research_backed=research_performed,
                finish_reason="stop",
                usage=usage_dict,
            )

        except asyncio.TimeoutError as exc:
            logger.warning("gemini_chat_timeout", model=self.model, timeout=self.timeout_seconds)
            raise ProviderTimeoutError(
                f"Gemini call timed out after {self.timeout_seconds}s.",
                provider=self.name,
                model=self.model,
            ) from exc

        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "resource_exhausted" in msg or "quota" in msg or "rate limit" in msg:
                logger.warning("gemini_rate_limited", model=self.model)
                raise RateLimitError(
                    f"Gemini rate limit or quota exceeded: {exc}",
                    provider=self.name,
                    model=self.model,
                    retry_after=60.0,
                ) from exc
            if "permission_denied" in msg or "unauthenticated" in msg or "api_key_invalid" in msg or "401" in msg or "403" in msg:
                logger.error("gemini_auth_error", model=self.model)
                raise AuthenticationError(
                    f"Gemini authentication failed: {exc}",
                    provider=self.name,
                    model=self.model,
                ) from exc
            if "invalid_argument" in msg or "400" in msg:
                logger.warning("gemini_invalid_argument", model=self.model)
                raise InvalidRequestError(
                    f"Gemini invalid request argument: {exc}",
                    provider=self.name,
                    model=self.model,
                ) from exc

            logger.error("gemini_provider_error", model=self.model, error=str(exc))
            raise ProviderUnavailableError(
                f"Gemini service unavailable: {exc}",
                provider=self.name,
                model=self.model,
            ) from exc
