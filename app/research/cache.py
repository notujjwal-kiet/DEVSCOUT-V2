"""
app/research/cache.py

Cache interface and in-memory implementation for web research discovery and page fetching.
Keys are normalized (queries and URLs) with configurable TTL from Settings.
"""
from __future__ import annotations

import time
from typing import Any, Protocol, runtime_checkable

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


@runtime_checkable
class ResearchCache(Protocol):
    """Protocol for research caches."""

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if missing or expired."""
        ...

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store a value with an optional TTL in seconds."""
        ...

    async def clear(self) -> None:
        """Clear all entries in the cache."""
        ...


class _CacheEntry:
    __slots__ = ("value", "expires_at")

    def __init__(self, value: Any, expires_at: float | None):
        self.value = value
        self.expires_at = expires_at

    def is_expired(self, now: float) -> bool:
        if self.expires_at is None:
            return False
        return now >= self.expires_at


class InMemoryResearchCache:
    """Thread-safe and async-friendly in-memory research cache with TTL support."""

    def __init__(self, default_ttl_seconds: int | None = None) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._default_ttl: int | None = default_ttl_seconds

    def _get_default_ttl(self) -> int:
        if self._default_ttl is not None:
            return self._default_ttl
        return get_settings().evidence_cache_ttl_seconds

    async def get(self, key: str) -> Any | None:
        """Retrieve key. If expired or missing, returns None."""
        now = time.monotonic()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            self._entries.pop(key, None)
            return None
        return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: int | None = None) -> None:
        """Store value with TTL."""
        ttl = ttl_seconds if ttl_seconds is not None else self._get_default_ttl()
        expires_at = time.monotonic() + ttl if ttl > 0 else None
        self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._entries.pop(key, None)

    async def clear(self) -> None:
        """Clear all cache entries."""
        self._entries.clear()


_global_cache: InMemoryResearchCache | None = None


def get_research_cache() -> InMemoryResearchCache:
    """Return the global in-memory research cache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = InMemoryResearchCache()
    return _global_cache


def reset_research_cache() -> None:
    """Reset the global cache instance (useful for test isolation)."""
    global _global_cache
    _global_cache = None
