"""
app/api/auth.py

Authentication and rate limiting dependencies for DEVSCOUT API.
- API Key verification via X-API-Key / Authorization header against settings.api_keys.
- In-memory token bucket rate limiter keyed by API key (or client IP).
"""
from __future__ import annotations

import asyncio
import time

from fastapi import HTTPException, Request, status

from app.config import get_settings


class TokenBucket:
    """In-memory token bucket algorithm for rate limiting."""

    def __init__(self, capacity: float, refill_rate: float) -> None:
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity
        self.last_refill = time.monotonic()

    def consume(self, amount: float = 1.0) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimiter:
    """Thread/async safe token bucket collection."""

    def __init__(self) -> None:
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    async def check(self, key: str, capacity: float, refill_rate: float) -> bool:
        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket(capacity, refill_rate)
                self._buckets[key] = bucket
            else:
                bucket.capacity = capacity
                bucket.refill_rate = refill_rate
            return bucket.consume(1.0)

    async def clear(self) -> None:
        async with self._lock:
            self._buckets.clear()


_GLOBAL_RATE_LIMITER = RateLimiter()


def get_rate_limiter() -> RateLimiter:
    """Return singleton rate limiter."""
    return _GLOBAL_RATE_LIMITER


async def verify_api_key_and_rate_limit(request: Request) -> str | None:
    """
    FastAPI dependency:
    1. Authenticate API key if cfg.api_keys is configured.
       Extracts from X-API-Key, api-key, or Authorization: Bearer <key>.
       Raises 401 Unauthorized if missing/invalid.
    2. Rate limit request using in-memory token bucket keyed by API key or IP.
       Raises 429 Too Many Requests if rate limit is exceeded.
    """
    cfg = get_settings()

    # Extract API key
    api_key = request.headers.get("x-api-key") or request.headers.get("api-key")
    if not api_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:].strip()

    valid_keys = [k.strip() for k in cfg.api_keys.split(",") if k.strip()]

    # Auth check
    if valid_keys:
        if not api_key or api_key not in valid_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing API key",
            )
        rate_key = f"key:{api_key}"
    else:
        # If no keys configured, key by provided api_key or client host
        if api_key:
            rate_key = f"key:{api_key}"
        else:
            host = request.client.host if request.client else "anonymous"
            rate_key = f"ip:{host}"

    # Rate limit check
    limiter = get_rate_limiter()
    allowed = await limiter.check(rate_key, cfg.rate_limit_capacity, cfg.rate_limit_refill_rate)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Try again later.",
        )

    return api_key
