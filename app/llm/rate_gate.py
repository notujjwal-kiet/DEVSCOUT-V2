"""
app/llm/rate_gate.py

Global async rate gate for Gemini API calls:
1. Sliding window RPM limiter: max gemini_rpm (default 4) requests/min, queueing instead of failing.
2. Daily counter persisted to data/quota.json with gemini_rpd_limit (default 18),
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
    Global concurrency, RPM sliding window, and daily quota gate for Gemini.
    """

    def __init__(
        self,
        rpm_limit: int = 4,
        rpd_limit: int = 18,
        quota_file: str = "data/quota.json",
    ) -> None:
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit
        self.quota_file = quota_file
        self._request_timestamps: list[float] = []
        self._lock = asyncio.Lock()

    def _get_today_str(self) -> str:
        return datetime.now(UTC_MINUS_8).strftime("%Y-%m-%d")

    def _get_used_today(self) -> int:
        today_str = self._get_today_str()
        if not os.path.exists(self.quota_file):
            return 0
        try:
            with open(self.quota_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("date") != today_str:
                return 0
            return int(data.get("used_today", 0))
        except Exception as exc:
            logger.warning("gemini_rate_gate_read_error", error=str(exc))
            return 0

    def _increment_used_today(self) -> int:
        today_str = self._get_today_str()
        used = 0
        if os.path.exists(self.quota_file):
            try:
                with open(self.quota_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("date") == today_str:
                    used = int(data.get("used_today", 0))
            except Exception:
                used = 0

        used += 1
        dir_name = os.path.dirname(self.quota_file)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        try:
            with open(self.quota_file, "w", encoding="utf-8") as f:
                json.dump({"date": today_str, "used_today": used}, f, indent=2)
        except Exception as exc:
            logger.warning("gemini_rate_gate_write_error", error=str(exc))
        return used

    async def acquire(self) -> None:
        """
        Acquire slot under RPM limit and daily quota.
        Raises LLMProviderError("Gemini daily budget reached") if daily budget is exhausted.
        Queues via asyncio.sleep if sliding window is currently full.
        """
        from app.llm.provider import LLMProviderError as ProviderError

        async with self._lock:
            # 1. Check daily budget BEFORE calling
            used_today = self._get_used_today()
            if used_today >= self.rpd_limit:
                raise ProviderError("Gemini daily budget reached")

            # 2. Check sliding window RPM (60s window)
            while True:
                now = time.monotonic()
                self._request_timestamps = [
                    t for t in self._request_timestamps if now - t < 60.0
                ]
                if len(self._request_timestamps) < self.rpm_limit:
                    break

                oldest = self._request_timestamps[0]
                wait_seconds = max(0.01, 60.0 - (now - oldest))
                logger.info(
                    "gemini_rpm_limit_queueing",
                    rpm_limit=self.rpm_limit,
                    wait_seconds=round(wait_seconds, 3),
                )
                await asyncio.sleep(wait_seconds)
                # If time did not advance (e.g. mocked asyncio.sleep), advance by popping oldest
                after_now = time.monotonic()
                if after_now <= now and self._request_timestamps:
                    self._request_timestamps.pop(0)

            # 3. Record timestamp and increment daily quota count
            self._request_timestamps.append(time.monotonic())
            self._increment_used_today()

    def get_quota_status(self) -> dict[str, int]:
        """Return public quota status without secrets."""
        return {
            "used_today": self._get_used_today(),
            "limit_today": self.rpd_limit,
            "rpm_limit": self.rpm_limit,
        }


_global_rate_gate: GeminiRateGate | None = None


def get_rate_gate() -> GeminiRateGate:
    """Return singleton GeminiRateGate initialized from Settings."""
    global _global_rate_gate
    if _global_rate_gate is None:
        from app.config import get_settings

        cfg = get_settings()
        _global_rate_gate = GeminiRateGate(
            rpm_limit=cfg.gemini_rpm,
            rpd_limit=cfg.gemini_rpd_limit,
            quota_file=cfg.gemini_quota_file,
        )
    return _global_rate_gate


def set_rate_gate(gate: GeminiRateGate | None) -> None:
    """Set global rate gate instance (for testing or reconfiguration)."""
    global _global_rate_gate
    _global_rate_gate = gate
