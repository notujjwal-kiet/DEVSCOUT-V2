"""
app/llm/rate_gate.py

Global async rate gate for Gemini API calls:
1. Per-model sliding window RPM limiter: max rpm requests/min (from registry or default),
   queueing instead of failing. Local models are unlimited.
2. Per-model daily counter persisted to data/quota.json with rpd limit (from registry or default),
   resetting at date change in UTC-8. Counts every request including retries.
3. Raises LLMProviderError("Gemini daily budget reached") BEFORE calling when daily budget exhausted.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from app.config import Settings

logger = structlog.get_logger(__name__)

UTC_MINUS_8 = timezone(timedelta(hours=-8))


class LLMProviderError(Exception):
    """Fallback LLMProviderError if imported before provider."""
    pass


class GeminiRateGate:
    """
    Global concurrency, per-model RPM sliding window, and daily quota gate.
    """

    def __init__(
        self,
        rpm_limit: int | None = None,
        rpd_limit: int | None = None,
        quota_file: str = "data/quota.json",
    ) -> None:
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self._explicit_rpm = rpm_limit
        self._explicit_rpd = rpd_limit
        self.quota_file = quota_file
        self._request_timestamps_by_model: dict[str, list[float]] = {}
        self._lock = asyncio.Lock()

    @property
    def _request_timestamps(self) -> list[float]:
        return self._request_timestamps_by_model.setdefault("gemini-main", [])

    @_request_timestamps.setter
    def _request_timestamps(self, val: list[float]) -> None:
        self._request_timestamps_by_model["gemini-main"] = val

    def _get_model_limits(self, model_id: str) -> tuple[int, int | None, bool]:
        """
        Return (rpm, rpd, is_unlimited).
        Local models are unlimited.
        Uses rpm/rpd from registry, or instance explicit overrides.
        """
        try:
            from app.llm.registry import get_model_registry

            reg = get_model_registry()
            entry = reg.get_model(model_id)
            if entry is not None:
                if entry.local or (entry.rpm is None and entry.rpd is None):
                    return (0, None, True)

                rpm = self._explicit_rpm if self._explicit_rpm is not None else (entry.rpm if entry.rpm is not None else 5)
                rpd = self._explicit_rpd if self._explicit_rpd is not None else (entry.rpd if entry.rpd is not None else 20)
                return (rpm, rpd, False)
        except Exception:
            pass

        rpm = self._explicit_rpm if self._explicit_rpm is not None else 5
        rpd = self._explicit_rpd if self._explicit_rpd is not None else 20
        return (rpm, rpd, False)

    def _get_today_str(self) -> str:
        return datetime.now(UTC_MINUS_8).strftime("%Y-%m-%d")

    def _get_used_today(self, model_id: str = "gemini-main") -> int:
        today_str = self._get_today_str()
        if not os.path.exists(self.quota_file):
            return 0
        try:
            with open(self.quota_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") != today_str:
                return 0
            models_data = data.get("models", {})
            if model_id in models_data:
                return int(models_data[model_id].get("used_today", 0))
            if model_id == "gemini-main" and "used_today" in data:
                return int(data.get("used_today", 0))
            return 0
        except Exception as exc:
            logger.warning("gemini_rate_gate_read_error", error=str(exc))
            return 0

    def _increment_used_today(self, model_id: str = "gemini-main") -> int:
        today_str = self._get_today_str()
        data = {}
        if os.path.exists(self.quota_file):
            try:
                with open(self.quota_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}

        if data.get("date") != today_str:
            data = {"date": today_str, "models": {}}

        models_data = data.setdefault("models", {})
        model_entry = models_data.setdefault(model_id, {})
        used = int(model_entry.get("used_today", 0)) + 1
        model_entry["used_today"] = used

        # Also maintain top-level used_today for gemini-main / backward compatibility
        if model_id == "gemini-main":
            data["used_today"] = used

        dir_name = os.path.dirname(self.quota_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            with open(self.quota_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            logger.warning("gemini_rate_gate_write_error", error=str(exc))
        return used

    def has_budget(self, model_id: str = "gemini-main") -> bool:
        """Return True if model has daily budget remaining (or is unlimited)."""
        rpm, rpd, is_unlimited = self._get_model_limits(model_id)
        if is_unlimited:
            return True
        if rpd is None:
            return True
        return self._get_used_today(model_id) < rpd

    async def acquire(self, model_id: str = "gemini-main") -> None:
        """
        Acquire slot under model RPM limit and daily quota.
        Raises LLMProviderError("Gemini daily budget reached") if daily budget is exhausted.
        Queues via asyncio.sleep if sliding window is currently full.
        Local models are unlimited.
        """
        from app.llm.provider import LLMProviderError as ProviderError

        rpm, rpd, is_unlimited = self._get_model_limits(model_id)
        if is_unlimited:
            return

        async with self._lock:
            # 1. Check daily budget BEFORE calling
            used_today = self._get_used_today(model_id)
            if rpd is not None and used_today >= rpd:
                raise ProviderError("Gemini daily budget reached")

            # 2. Check sliding window RPM (60s window) for this model
            timestamps = self._request_timestamps_by_model.setdefault(model_id, [])
            while True:
                now = time.monotonic()
                timestamps = [t for t in timestamps if now - t < 60.0]
                self._request_timestamps_by_model[model_id] = timestamps
                if len(timestamps) < rpm:
                    break

                oldest = timestamps[0]
                wait_seconds = max(0.01, 60.0 - (now - oldest))
                logger.info(
                    "gemini_rpm_limit_queueing",
                    model_id=model_id,
                    rpm_limit=rpm,
                    wait_seconds=round(wait_seconds, 3),
                )
                await asyncio.sleep(wait_seconds)
                after_now = time.monotonic()
                if after_now <= now and timestamps:
                    timestamps.pop(0)

            # 3. Record timestamp and increment daily quota count
            timestamps.append(time.monotonic())
            self._increment_used_today(model_id)

    def get_quota_status(self, model_id: str | None = None) -> dict[str, int | None]:
        """Return public quota status without secrets for given model (or gemini-main)."""
        m_id = model_id or "gemini-main"
        rpm, rpd, is_unlimited = self._get_model_limits(m_id)
        if is_unlimited:
            return {
                "used_today": 0,
                "limit_today": None,
                "rpm_limit": None,
            }
        return {
            "used_today": self._get_used_today(m_id),
            "limit_today": rpd,
            "rpm_limit": rpm,
        }


_global_rate_gate: GeminiRateGate | None = None


def get_rate_gate() -> GeminiRateGate:
    """Return singleton GeminiRateGate initialized from Settings."""
    global _global_rate_gate
    if _global_rate_gate is None:
        from app.config import get_settings

        cfg = get_settings()
        _global_rate_gate = GeminiRateGate(
            rpm_limit=None,
            rpd_limit=None,
            quota_file=cfg.gemini_quota_file,
        )
    return _global_rate_gate


def set_rate_gate(gate: GeminiRateGate | None) -> None:
    """Set global rate gate instance (for testing or reconfiguration)."""
    global _global_rate_gate
    _global_rate_gate = gate
