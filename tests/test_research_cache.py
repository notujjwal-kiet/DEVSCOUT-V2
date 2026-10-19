"""
tests/test_research_cache.py

Unit tests for research cache interface and in-memory TTL implementation:
- InMemoryResearchCache basic operations, TTL expiration, and default TTL from settings.
- discover_sources caching by normalized query.
- fetch_page caching by normalized URL.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock
import httpx
import pytest
from google.genai import types

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import ResourceType
from app.research.cache import (
    InMemoryResearchCache,
    ResearchCache,
    get_research_cache,
    reset_research_cache,
)
from app.research.discovery import discover_sources, normalize_query
from app.research.models import DiscoveredSource, ResearchDocument
from app.research.web_fetch import fetch_page, normalize_url


@pytest.mark.asyncio
async def test_in_memory_cache_protocol_and_crud():
    """InMemoryResearchCache implements ResearchCache protocol with get, set, delete, and clear."""
    cache = InMemoryResearchCache(default_ttl_seconds=60)
    assert isinstance(cache, ResearchCache)

    # get on missing key
    assert await cache.get("missing") is None

    # set and get
    await cache.set("test-key", {"data": 123})
    assert await cache.get("test-key") == {"data": 123}

    # delete
    await cache.delete("test-key")
    assert await cache.get("test-key") is None

    # clear
    await cache.set("k1", "v1")
    await cache.set("k2", "v2")
    await cache.clear()
    assert await cache.get("k1") is None
    assert await cache.get("k2") is None


@pytest.mark.asyncio
async def test_in_memory_cache_ttl_expiration():
    """Cache entry expires after specified TTL in seconds."""
    cache = InMemoryResearchCache(default_ttl_seconds=1)

    # Set with short TTL
    await cache.set("expiring-key", "val", ttl_seconds=0.05)
    assert await cache.get("expiring-key") == "val"

    await asyncio.sleep(0.08)
    assert await cache.get("expiring-key") is None


@pytest.mark.asyncio
async def test_cache_reads_default_ttl_from_settings(monkeypatch):
    """Default TTL is read dynamically from get_settings().evidence_cache_ttl_seconds."""
    monkeypatch.setenv("EVIDENCE_CACHE_TTL_SECONDS", "3600")
    get_settings.cache_clear()
    cache = InMemoryResearchCache()
    assert cache._get_default_ttl() == 3600


@pytest.mark.asyncio
async def test_discover_sources_caching_by_normalized_query(monkeypatch):
    """discover_sources caches results by normalized query and avoids repeat searches."""
    reset_research_cache()

    chunk = types.GroundingChunk(
        web=types.GroundingChunkWeb(
            uri="https://fastapi.tiangolo.com/tutorial/",
            title="FastAPI Tutorial",
            domain="fastapi.tiangolo.com",
        )
    )
    mock_resp = types.GenerateContentResponse(
        candidates=[types.Candidate(grounding_metadata=types.GroundingMetadata(grounding_chunks=[chunk]))]
    )

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    async def mock_resolve(url, max_redirects=3, client=None):
        return url

    monkeypatch.setattr("app.research.discovery.resolve_redirects", mock_resolve)

    # First call: cache miss, provider called
    res1 = await discover_sources("  FastAPI   Tutorial  ", max_results=5, provider=provider)
    assert len(res1) == 1
    assert provider.generate_with_search.call_count == 1

    # Second call with variation of whitespace/casing: cache hit, provider NOT called
    res2 = await discover_sources("fastapi tutorial", max_results=5, provider=provider)
    assert len(res2) == 1
    assert res2[0].url == res1[0].url
    assert provider.generate_with_search.call_count == 1


@pytest.mark.asyncio
async def test_fetch_page_caching_by_normalized_url(monkeypatch):
    """fetch_page caches parsed ResearchDocument by normalized URL and avoids repeat HTTP requests."""
    reset_research_cache()

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        html = "<html><head><title>Test Doc</title></head><body><main><p>Content paragraph</p></main></body></html>"
        return httpx.Response(200, text=html, headers={"content-type": "text/html"})

    transport = httpx.MockTransport(handler)

    async def mock_resolve(url, max_redirects=3, client=None):
        return url

    monkeypatch.setattr("app.research.web_fetch.resolve_redirects", mock_resolve)

    async with httpx.AsyncClient(transport=transport) as client:
        # First fetch: triggers HTTP stream
        doc1 = await fetch_page(
            url="https://example.com/docs/?utm_source=twitter&utm_medium=social#overview",
            resource_type=ResourceType.DOCS,
            client=client,
        )
        assert doc1.resource.title == "Test Doc"
        assert call_count == 1

        # Second fetch with tracking param / fragment variation: normalized URL matches cache, skips HTTP stream
        doc2 = await fetch_page(
            url="https://example.com/docs/?gclid=123",
            resource_type=ResourceType.DOCS,
            client=client,
        )
        assert doc2.resource.title == "Test Doc"
        assert doc2.resource.content_hash == doc1.resource.content_hash
        assert call_count == 1
