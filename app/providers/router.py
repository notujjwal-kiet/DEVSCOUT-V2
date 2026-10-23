"""
app/providers/router.py

Intelligent task-based routing and fallback across AI providers:
- Selects provider based on task (research -> Gemini, chat/code/explain -> Groq/Cerebras/Ollama).
- Falls back on eligible transient failures (rate limits, 5xx outages, timeouts).
- Respects Retry-After backoff and avoids hammering exhausted quotas.
- Never retries authentication or invalid-request errors against the same provider.
- Never logs API keys or leaks credentials.
"""
from __future__ import annotations

import time
from typing import Any
import structlog

from app.providers.base import (
    AuthenticationError,
    BaseChatProvider,
    ChatMessage,
    ChatResponse,
    InvalidRequestError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from app.providers.registry import ProviderRegistry, get_provider_registry

logger = structlog.get_logger(__name__)


class ProviderRouter:
    """
    Routes chat and research requests to appropriate providers with intelligent fallback.
    """

    def __init__(self, registry: ProviderRegistry | None = None) -> None:
        self.registry = registry or get_provider_registry()
        # Cooldown tracking: provider_name -> unix timestamp until which provider is in cooldown
        self._cooldowns: dict[str, float] = {}

    def _is_in_cooldown(self, provider_name: str) -> bool:
        """Check if provider is temporarily suspended due to rate limit or quota."""
        expiry = self._cooldowns.get(provider_name, 0.0)
        return time.time() < expiry

    def _set_cooldown(self, provider_name: str, seconds: float) -> None:
        """Set backoff cooldown for a provider."""
        self._cooldowns[provider_name] = time.time() + max(1.0, seconds)
        logger.warning(
            "provider_cooldown_set",
            provider=provider_name,
            cooldown_seconds=seconds,
        )

    def select_candidate_chain(
        self,
        task: str = "chat",
        preferred_model_or_provider: str | None = None,
    ) -> list[BaseChatProvider]:
        """
        Build an ordered list of candidate providers based on task and user preferences.
        """
        all_providers = {p.name: p for p in self.registry.list_providers() if p.is_available()}
        candidates: list[BaseChatProvider] = []

        # If user explicitly preferred a specific provider/model
        preferred_name: str | None = None
        if preferred_model_or_provider:
            pref = preferred_model_or_provider.lower().strip()
            for name, prov in all_providers.items():
                if name == pref or pref.startswith(name) or prov.model.lower() == pref:
                    preferred_name = name
                    candidates.append(prov)
                    break

        if task == "research":
            # For research tasks: Gemini is the primary research provider
            chain = ["gemini", "groq", "cerebras", "ollama"]
        else:
            # For standard chat, code, and explanation tasks
            chain = ["groq", "cerebras", "ollama", "gemini"]

        for name in chain:
            if name != preferred_name and name in all_providers:
                candidates.append(all_providers[name])

        return candidates

    async def route_chat(
        self,
        messages: list[ChatMessage],
        task: str = "chat",
        preferred_model_or_provider: str | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """
        Execute chat request through the candidate chain with automatic fallback.
        """
        candidates = self.select_candidate_chain(
            task=task,
            preferred_model_or_provider=preferred_model_or_provider,
        )

        if not candidates:
            # Fall back to any configured provider even if not initially flagged available
            all_list = self.registry.list_providers()
            if not all_list:
                raise ProviderUnavailableError(
                    "No AI providers are configured on the backend. Please check .env configuration."
                )
            candidates = all_list

        last_error: Exception | None = None
        attempted_providers: list[str] = []

        for provider in candidates:
            provider_name = provider.name

            # Skip provider if currently in cooldown (e.g. active 429 quota exhaustion)
            # unless it is the only remaining option
            if self._is_in_cooldown(provider_name) and len(candidates) > 1:
                logger.info(
                    "provider_skipped_in_cooldown",
                    provider=provider_name,
                    task=task,
                )
                continue

            attempted_providers.append(provider_name)
            logger.info(
                "provider_attempting_request",
                provider=provider_name,
                model=provider.model,
                task=task,
            )

            try:
                response = await provider.chat(messages=messages, task=task, **kwargs)

                # Ensure task metadata is accurate
                response.task = task

                # If task was research but fell back to a non-research provider without live research
                if task == "research" and provider_name != "gemini" and not kwargs.get("research_performed"):
                    response.research_backed = False

                logger.info(
                    "provider_request_succeeded",
                    provider=response.provider,
                    model=response.model,
                    task=task,
                    research_backed=response.research_backed,
                )
                return response

            except RateLimitError as exc:
                # 429: Respect Retry-After and record cooldown, then fall back
                backoff = exc.retry_after or 60.0
                self._set_cooldown(provider_name, backoff)
                logger.warning(
                    "provider_rate_limited_falling_back",
                    provider=provider_name,
                    retry_after=backoff,
                    error=str(exc),
                )
                last_error = exc
                continue

            except (ProviderTimeoutError, ProviderUnavailableError) as exc:
                # Transient error: Log and fall back to next provider
                logger.warning(
                    "provider_transient_failure_falling_back",
                    provider=provider_name,
                    error=str(exc),
                )
                last_error = exc
                continue

            except (AuthenticationError, InvalidRequestError) as exc:
                # Auth or Bad Request: DO NOT retry against the same provider
                logger.error(
                    "provider_permanent_error_skipping_provider",
                    provider=provider_name,
                    error=str(exc),
                )
                last_error = exc
                # Fall back to next provider in chain
                continue

            except Exception as exc:
                logger.error(
                    "provider_unexpected_error",
                    provider=provider_name,
                    error=str(exc),
                )
                last_error = exc
                continue

        # All candidate providers failed
        err_msg = (
            f"All candidate providers ({', '.join(attempted_providers)}) failed. "
            f"Last error: {last_error}"
        )
        logger.error("all_providers_exhausted", attempted=attempted_providers, last_error=str(last_error))
        raise ProviderUnavailableError(err_msg, provider="router")


_router_instance: ProviderRouter | None = None


def get_provider_router() -> ProviderRouter:
    """Return singleton ProviderRouter instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = ProviderRouter()
    return _router_instance
