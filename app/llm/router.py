"""
app/llm/router.py

Dynamic provider routing and failover across registered models:
- Validates requested_id against registry (raises HTTP 422 if unknown).
- If requested_id is given: used alone, with Ollama fallback only if LLM_FALLBACK=ollama.
- In auto mode: choose the lowest-priority-number model that has budget left.
- When a call fails with quota/rate/daily budget, moves to the next model once.
- Grounded search (discovery) only runs on Gemini models that have budget left.
"""
from __future__ import annotations

import os
from typing import TypeVar
from fastapi import HTTPException
import structlog

from app.llm.provider import LLMProvider, RateLimitError, LLMProviderError, get_llm_provider, GeminiProvider
from app.llm.rate_gate import get_rate_gate
from app.llm.registry import ModelEntry, get_model_registry

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _is_quota_or_rate_error(exc: Exception) -> bool:
    """Check if exception represents a quota exhaustion, daily budget limit, or 429."""
    if isinstance(exc, RateLimitError):
        return True
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True
    msg = str(exc).lower()
    return (
        "daily budget reached" in msg
        or "budget reached" in msg
        or "quota" in msg
        or "rate limit" in msg
        or "rate_limit" in msg
        or "resource_exhausted" in msg
        or "429" in msg
    )


class RoutedProvider:
    """
    Provider wrapper that routes LLM requests to a primary model and
    fails over to the next candidate model once on quota/rate/daily budget errors.
    """

    def __init__(
        self,
        primary_entry: ModelEntry,
        fallback_entry: ModelEntry | None = None,
        stage: str | int | None = None,
        first_choice_model_id: str | None = None,
    ) -> None:
        self.primary_entry = primary_entry
        self.fallback_entry = fallback_entry
        self.stage = stage
        self.first_choice_model_id = first_choice_model_id or primary_entry.id
        self.active_entry = primary_entry
        self.model_id: str = primary_entry.id
        self.model_name: str = primary_entry.model
        self.was_fallback: bool = (self.model_id != self.first_choice_model_id)

        self._primary_instance: LLMProvider | None = None
        self._fallback_instance: LLMProvider | None = None

        self.last_usage_metadata: object | None = None
        self.last_usage_data: object | None = None

    def _get_provider_instance(self, entry: ModelEntry) -> LLMProvider:
        p = get_llm_provider(
            provider_name=entry.provider,
            model_name=entry.model,
            model_id=entry.id,
            rate_gate=get_rate_gate(),
        )
        p.model_id = entry.id  # type: ignore[attr-defined]
        return p

    def _get_primary(self) -> LLMProvider:
        if self._primary_instance is None:
            self._primary_instance = self._get_provider_instance(self.primary_entry)
        return self._primary_instance

    def _get_fallback(self) -> LLMProvider:
        if self._fallback_instance is None and self.fallback_entry is not None:
            self._fallback_instance = self._get_provider_instance(self.fallback_entry)
        if self._fallback_instance is None:
            raise RuntimeError("No fallback provider configured")
        return self._fallback_instance

    def _sync_telemetry(self, provider: LLMProvider) -> None:
        self.last_usage_metadata = getattr(provider, "last_usage_metadata", None)
        self.last_usage_data = getattr(provider, "last_usage_data", None)

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
        provider = self._get_primary()
        try:
            res = await provider.generate(prompt, system_prompt=system_prompt, **kwargs)
            self._sync_telemetry(provider)
            return res
        except Exception as exc:
            if _is_quota_or_rate_error(exc) and self.fallback_entry:
                logger.warning(
                    "router_failover_on_quota",
                    stage=self.stage,
                    from_model=self.primary_entry.id,
                    to_model=self.fallback_entry.id,
                    error=str(exc),
                )
                self.was_fallback = True
                self.active_entry = self.fallback_entry
                self.model_id = self.fallback_entry.id
                self.model_name = self.fallback_entry.model
                fb = self._get_fallback()
                res = await fb.generate(prompt, system_prompt=system_prompt, **kwargs)
                self._sync_telemetry(fb)
                return res
            raise

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: object
    ) -> T:
        provider = self._get_primary()
        try:
            res = await provider.generate_structured(prompt, schema=schema, system_prompt=system_prompt, **kwargs)
            self._sync_telemetry(provider)
            return res
        except Exception as exc:
            if _is_quota_or_rate_error(exc) and self.fallback_entry:
                logger.warning(
                    "router_failover_on_quota",
                    stage=self.stage,
                    from_model=self.primary_entry.id,
                    to_model=self.fallback_entry.id,
                    error=str(exc),
                )
                self.was_fallback = True
                self.active_entry = self.fallback_entry
                self.model_id = self.fallback_entry.id
                self.model_name = self.fallback_entry.model
                fb = self._get_fallback()
                res = await fb.generate_structured(prompt, schema=schema, system_prompt=system_prompt, **kwargs)
                self._sync_telemetry(fb)
                return res
            raise

    async def generate_with_search(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> object:
        provider = self._get_primary()
        if not hasattr(provider, "generate_with_search"):
            raise LLMProviderError(f"Model '{self.model_id}' does not support grounded search")
        try:
            res = await provider.generate_with_search(prompt, system_prompt=system_prompt, **kwargs)  # type: ignore[attr-defined]
            self._sync_telemetry(provider)
            return res
        except Exception as exc:
            if _is_quota_or_rate_error(exc) and self.fallback_entry and self.fallback_entry.provider == "gemini":
                logger.warning(
                    "router_search_failover_on_quota",
                    stage=self.stage,
                    from_model=self.primary_entry.id,
                    to_model=self.fallback_entry.id,
                    error=str(exc),
                )
                self.was_fallback = True
                self.active_entry = self.fallback_entry
                self.model_id = self.fallback_entry.id
                self.model_name = self.fallback_entry.model
                fb = self._get_fallback()
                if hasattr(fb, "generate_with_search"):
                    res = await fb.generate_with_search(prompt, system_prompt=system_prompt, **kwargs)  # type: ignore[attr-defined]
                    self._sync_telemetry(fb)
                    return res
            raise


def can_run_grounded_search(provider: object) -> bool:
    """
    Check if discovery (grounded search) can run:
    Discovery only runs on Gemini models that have budget; otherwise use seed docs and GitHub.
    """
    if provider is None:
        try:
            reg = get_model_registry()
            gate = get_rate_gate()
            for m in reg.list_models():
                if m.provider == "gemini" and gate.has_budget(m.id):
                    return True
        except Exception:
            pass
        return False

    # Check for unit test mocks/fakes
    if hasattr(provider, "generate_with_search") and not isinstance(provider, (RoutedProvider, GeminiProvider)):
        return True

    active_entry = getattr(provider, "active_entry", None)
    if active_entry is not None:
        if active_entry.provider != "gemini":
            return False
        return get_rate_gate().has_budget(active_entry.id)

    if isinstance(provider, GeminiProvider):
        model_id = getattr(provider, "model_id", "gemini-main")
        return get_rate_gate().has_budget(model_id)

    return False


async def get_provider_for(
    stage: str | int | None = None,
    requested_id: str | None = None,
) -> LLMProvider:
    """
    Get provider for pipeline stage:
    - If requested_id is given: must exist in registry (else 422), used alone,
      with Ollama fallback only if LLM_FALLBACK=ollama.
    - In auto mode: choose lowest-priority-number model with budget left;
      when a call fails with quota/rate/daily budget, move to the next model once.
    """
    reg = get_model_registry()
    rate_gate = get_rate_gate()

    # 1. Explicit model request
    if requested_id is not None:
        entry = reg.get_model(requested_id)
        if entry is None:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown model_id '{requested_id}'. Available models: {reg.get_model_ids()}",
            )

        fallback_entry: ModelEntry | None = None
        # Ollama fallback only if LLM_FALLBACK=ollama
        fb_env = os.getenv("LLM_FALLBACK", "").lower().strip()
        if fb_env == "ollama":
            for m in reg.list_models():
                if m.provider == "ollama" and m.id != entry.id:
                    fallback_entry = m
                    break

        return RoutedProvider(
            primary_entry=entry,
            fallback_entry=fallback_entry,
            stage=stage,
            first_choice_model_id=entry.id,
        )

    # 2. Auto mode: choose lowest-priority-number model with budget left
    models = reg.list_models()  # sorted by priority ascending
    first_choice_model_id = models[0].id if models else None

    # Filter to models that have budget left (local models always have budget)
    available_models = [m for m in models if rate_gate.has_budget(m.id)]

    if not available_models:
        # If remote models exhausted, fallback to first local/ollama model
        local_models = [m for m in models if m.local or m.provider == "ollama"]
        primary_entry = local_models[0] if local_models else models[0]
        fallback_entry = None
    else:
        primary_entry = available_models[0]
        # Next candidate model by priority
        remaining = [m for m in models if m.id != primary_entry.id]
        fallback_entry = remaining[0] if remaining else None

    return RoutedProvider(
        primary_entry=primary_entry,
        fallback_entry=fallback_entry,
        stage=stage,
        first_choice_model_id=first_choice_model_id,
    )
