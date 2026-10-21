"""
app/research/cache.py

Cache interface, persistent JSON file-backed implementation, and in-memory cache
for web research discovery and page fetching.
Keys are normalized (queries and URLs) with configurable TTL from Settings.
Entries are persisted to a JSON file under data/ so cached discovery and
fetched pages survive server restarts.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)


@runtime_checkable
class ResearchCache(Protocol):
    """Protocol for research caches."""

    async def get(self, key: str) -> Any | None:
        """Retrieve a value by key. Returns None if missing or expired."""
        ...

    async def set(self, key: str, value: Any, ttl_seconds: float | int | None = None) -> None:
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


def _serialize_value(val: Any) -> Any:
    """Serialize Pydantic models and structures for JSON persistence."""
    if isinstance(val, BaseModel):
        return {"__type__": val.__class__.__name__, "__data__": val.model_dump(mode="json")}
    elif isinstance(val, list):
        if val and isinstance(val[0], BaseModel):
            item_type = val[0].__class__.__name__
            return {"__type__": f"list[{item_type}]", "__data__": [item.model_dump(mode="json") for item in val]}
        return [_serialize_value(item) for item in val]
    elif isinstance(val, dict):
        return {k: _serialize_value(v) for k, v in val.items()}
    return val


def _deserialize_value(raw: Any) -> Any:
    """Deserialize Pydantic models from persisted JSON."""
    if isinstance(raw, dict):
        t = raw.get("__type__")
        d = raw.get("__data__")
        if t == "DiscoveredSource" and isinstance(d, dict):
            from app.research.models import DiscoveredSource

            return DiscoveredSource.model_validate(d)
        elif t == "ResearchDocument" and isinstance(d, dict):
            from app.research.models import ResearchDocument

            return ResearchDocument.model_validate(d)
        elif t == "list[DiscoveredSource]" and isinstance(d, list):
            from app.research.models import DiscoveredSource

            return [DiscoveredSource.model_validate(x) for x in d]
        elif t == "list[ResearchDocument]" and isinstance(d, list):
            from app.research.models import ResearchDocument

            return [ResearchDocument.model_validate(x) for x in d]
        return {k: _deserialize_value(v) for k, v in raw.items()}
    elif isinstance(raw, list):
        return [_deserialize_value(item) for item in raw]
    return raw


class InMemoryResearchCache:
    """
    Thread-safe, async-friendly research cache with TTL and persistent JSON file storage.
    Persists to data/research_cache.json so discovery and fetched pages survive server restarts.
    """

    def __init__(
        self,
        default_ttl_seconds: int | None = None,
        file_path: str | None = None,
    ) -> None:
        self._entries: dict[str, _CacheEntry] = {}
        self._default_ttl: int | None = default_ttl_seconds

        cfg = None
        try:
            cfg = get_settings()
        except Exception:
            pass

        if file_path is not None:
            self._file_path = file_path
        elif cfg and hasattr(cfg, "research_cache_file"):
            self._file_path = cfg.research_cache_file
        else:
            self._file_path = "data/research_cache.json"

        self._load_from_file()

    def _get_default_ttl(self) -> int:
        if self._default_ttl is not None:
            return self._default_ttl
        return get_settings().evidence_cache_ttl_seconds

    def _load_from_file(self) -> None:
        """Load persisted cache entries from JSON file, dropping expired entries."""
        if not self._file_path or not os.path.exists(self._file_path):
            return
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            now = time.time()
            if isinstance(raw_data, dict):
                for k, item in raw_data.items():
                    if isinstance(item, dict):
                        expires_at = item.get("expires_at")
                        if expires_at is not None and now >= expires_at:
                            continue
                        val = _deserialize_value(item.get("value"))
                        self._entries[k] = _CacheEntry(value=val, expires_at=expires_at)
        except Exception as exc:
            logger.warning("research_cache_load_failed", file_path=self._file_path, error=str(exc))

    def _save_to_file(self) -> None:
        """Save non-expired entries to JSON file."""
        if not self._file_path:
            return
        try:
            dir_name = os.path.dirname(self._file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            now = time.time()
            data_to_save: dict[str, Any] = {}
            for k, entry in list(self._entries.items()):
                if not entry.is_expired(now):
                    data_to_save[k] = {
                        "value": _serialize_value(entry.value),
                        "expires_at": entry.expires_at,
                    }
            tmp_path = f"{self._file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2)
            os.replace(tmp_path, self._file_path)
        except Exception as exc:
            logger.warning("research_cache_save_failed", file_path=self._file_path, error=str(exc))

    async def get(self, key: str) -> Any | None:
        """Retrieve key. If expired or missing, returns None."""
        now = time.time()
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.is_expired(now):
            self._entries.pop(key, None)
            self._save_to_file()
            return None
        return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: float | int | None = None) -> None:
        """Store value with TTL and persist to file."""
        ttl = ttl_seconds if ttl_seconds is not None else self._get_default_ttl()
        expires_at = time.time() + ttl if ttl > 0 else None
        self._entries[key] = _CacheEntry(value=value, expires_at=expires_at)
        self._save_to_file()

    async def delete(self, key: str) -> None:
        """Delete key from cache and update file."""
        if key in self._entries:
            self._entries.pop(key, None)
            self._save_to_file()

    async def clear(self) -> None:
        """Clear all cache entries and file."""
        self._entries.clear()
        self._save_to_file()


_global_cache: InMemoryResearchCache | None = None


def get_research_cache() -> InMemoryResearchCache:
    """Return the global research cache singleton."""
    global _global_cache
    if _global_cache is None:
        _global_cache = InMemoryResearchCache()
    return _global_cache


def reset_research_cache() -> None:
    """Reset the global cache instance (useful for test isolation)."""
    global _global_cache
    if _global_cache is not None:
        _global_cache._entries.clear()
        try:
            if os.path.exists(_global_cache._file_path):
                os.remove(_global_cache._file_path)
        except Exception:
            pass
    _global_cache = None
