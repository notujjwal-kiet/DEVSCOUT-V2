"""
tests/test_research_discovery.py

Unit tests for app.research.discovery and app.research.web_fetch:
- Normal search-grounding discovery
- Duplicate URL deduplication
- Redirect resolution (following redirect to final target)
- Failed redirect / SSRF dropped gracefully
- Missing grounding metadata handled without crashing
- Source type hint inference across domains and paths
- SSRF URL validation constraints
"""
import asyncio
import socket
from unittest.mock import AsyncMock
import httpx
import pytest
from google.genai import types

from app.llm.provider import GeminiProvider
from app.models.pipeline import ResourceType
from app.research.discovery import (
    DISCOVERY_CONCURRENCY,
    DOMAIN_TYPE_MAPPING,
    discover_sources,
    infer_source_type_hint,
    normalize_url,
)
from app.research.web_fetch import (
    SSRFError,
    WebFetchError,
    is_safe_url,
    resolve_and_validate,
    resolve_redirects,
)


def _make_mock_response(chunks: list[types.GroundingChunk] | None) -> types.GenerateContentResponse:
    """Helper to build a typed GenerateContentResponse with fake grounding metadata."""
    if chunks is None:
        cand = types.Candidate(grounding_metadata=None)
        return types.GenerateContentResponse(candidates=[cand])
    meta = types.GroundingMetadata(grounding_chunks=chunks)
    cand = types.Candidate(grounding_metadata=meta)
    return types.GenerateContentResponse(candidates=[cand])


def _make_web_chunk(uri: str, title: str, domain: str) -> types.GroundingChunk:
    """Helper to build a GroundingChunk with a GroundingChunkWeb payload."""
    web = types.GroundingChunkWeb(uri=uri, title=title, domain=domain)
    return types.GroundingChunk(web=web)


# ── 1. Normal Discovery Case ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sources_normal_case():
    """Verify standard discovery parses grounding chunks into DiscoveredSource items."""
    chunks = [
        _make_web_chunk("https://fastapi.tiangolo.com/tutorial/", "FastAPI Tutorial", "fastapi.tiangolo.com"),
        _make_web_chunk("https://github.com/tiangolo/fastapi", "GitHub - tiangolo/fastapi", "github.com"),
        _make_web_chunk("https://news.ycombinator.com/item?id=123", "HN Discussion", "news.ycombinator.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    # HTTP client returns 200 for each URL
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        sources = await discover_sources(
            query="FastAPI tutorials",
            max_results=5,
            provider=provider,
            http_client=http_client,
        )

    assert len(sources) == 3
    assert sources[0].url == "https://fastapi.tiangolo.com/tutorial/"
    assert sources[0].source_type_hint == ResourceType.DOCS
    assert sources[1].url == "https://github.com/tiangolo/fastapi"
    assert sources[1].source_type_hint == ResourceType.REPO
    assert sources[2].url == "https://news.ycombinator.com/item?id=123"
    assert sources[2].source_type_hint == ResourceType.COMMUNITY


# ── 2. Duplicate URLs Deduplication ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sources_deduplicates_urls():
    """Verify duplicate final URLs are deduplicated while preserving order."""
    chunks = [
        _make_web_chunk("https://fastapi.tiangolo.com", "FastAPI Docs", "fastapi.tiangolo.com"),
        _make_web_chunk("https://fastapi.tiangolo.com", "FastAPI Docs Duplicate", "fastapi.tiangolo.com"),
        _make_web_chunk("https://github.com/tiangolo/fastapi", "GitHub Repo", "github.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        sources = await discover_sources(
            query="FastAPI",
            max_results=5,
            provider=provider,
            http_client=http_client,
        )

    assert len(sources) == 2
    assert sources[0].url == "https://fastapi.tiangolo.com"
    assert sources[1].url == "https://github.com/tiangolo/fastapi"


# ── 3. Redirect Resolution ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sources_resolves_redirects():
    """Verify redirect URIs are followed and store the terminal target URL."""
    redirect_uri = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/ABC123"
    final_target = "https://fastapi.tiangolo.com/security/"

    chunks = [
        _make_web_chunk(redirect_uri, "FastAPI Security", "vertexaisearch.cloud.google.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://vertexaisearch.cloud.google.com"):
            return httpx.Response(302, headers={"Location": final_target})
        return httpx.Response(200, text="Final content")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        sources = await discover_sources(
            query="FastAPI auth",
            max_results=5,
            provider=provider,
            http_client=http_client,
        )

    assert len(sources) == 1
    assert sources[0].url == final_target
    assert sources[0].domain == "fastapi.tiangolo.com"
    assert sources[0].source_type_hint == ResourceType.DOCS


# ── 4. Failed Redirect Dropped ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sources_drops_failed_redirects():
    """Verify that sources whose redirect resolution fails (e.g. 404 or SSRF) are dropped without crashing."""
    broken_uri = "https://broken-link.example.com/missing"
    valid_uri = "https://github.com/psf/requests"

    chunks = [
        _make_web_chunk(broken_uri, "Broken", "broken-link.example.com"),
        _make_web_chunk(valid_uri, "Requests Repo", "github.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    def handler(request: httpx.Request) -> httpx.Response:
        if "broken-link" in str(request.url):
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as http_client:
        sources = await discover_sources(
            query="Requests library",
            max_results=5,
            provider=provider,
            http_client=http_client,
        )

    assert len(sources) == 1
    assert sources[0].url == valid_uri
    assert sources[0].source_type_hint == ResourceType.REPO


# ── 5. Missing Grounding Metadata ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_discover_sources_no_grounding_metadata():
    """Verify missing or empty grounding metadata returns [] and never crashes."""
    mock_resp = _make_mock_response(None)
    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    sources = await discover_sources(
        query="Some obscure query with no web search results",
        max_results=5,
        provider=provider,
    )
    assert sources == []


# ── 6. Source Type Hint Mapping ───────────────────────────────────────────────

def test_source_type_hint_mappings():
    """Verify domain and path-based source type hint classifications."""
    assert infer_source_type_hint("https://github.com/tiangolo/fastapi", "github.com") == ResourceType.REPO
    assert infer_source_type_hint("https://gist.github.com/user/123", "gist.github.com") == ResourceType.REPO

    assert infer_source_type_hint("https://stackoverflow.com/questions/123", "stackoverflow.com") == ResourceType.COMMUNITY
    assert infer_source_type_hint("https://reddit.com/r/python", "reddit.com") == ResourceType.COMMUNITY
    assert infer_source_type_hint("https://news.ycombinator.com/item?id=456", "news.ycombinator.com") == ResourceType.COMMUNITY

    assert infer_source_type_hint("https://arxiv.org/abs/2301.00000", "arxiv.org") == ResourceType.PAPER

    assert infer_source_type_hint("https://docs.python.org/3/library/asyncio.html", "docs.python.org") == ResourceType.DOCS
    assert infer_source_type_hint("https://fastapi.tiangolo.com/tutorial/", "fastapi.tiangolo.com") == ResourceType.DOCS
    assert infer_source_type_hint("https://example.org/docs/getting-started", "example.org") == ResourceType.DOCS

    assert infer_source_type_hint("https://techblog.netflix.com/microservices", "techblog.netflix.com") == ResourceType.BLOG
    assert infer_source_type_hint("https://unknown-domain.io/article", "unknown-domain.io") == ResourceType.BLOG


# ── 7. SSRF Safety Validation ─────────────────────────────────────────────────

def test_ssrf_safety_rules():
    """Verify is_safe_url prevents SSRF vulnerabilities."""
    # Safe URLs
    assert is_safe_url("https://fastapi.tiangolo.com") is True
    assert is_safe_url("http://github.com/user/repo") is True
    assert is_safe_url("https://docs.python.org") is True

    # Unsafe schemes
    assert is_safe_url("file:///etc/passwd") is False
    assert is_safe_url("ftp://ftp.example.com") is False
    assert is_safe_url("gopher://localhost:70") is False

    # Unsafe hostnames (Loopback & Local)
    assert is_safe_url("http://localhost:8000") is False
    assert is_safe_url("http://127.0.0.1:5000") is False
    assert is_safe_url("http://0.0.0.0:80") is False
    assert is_safe_url("http://test.localhost") is False
    assert is_safe_url("http://internal.local") is False

    # Cloud metadata endpoints
    assert is_safe_url("http://169.254.169.254/latest/meta-data/") is False
    assert is_safe_url("http://metadata.google.internal/computeMetadata/v1/") is False

    # Private IP ranges
    assert is_safe_url("http://10.0.0.1/admin") is False
    assert is_safe_url("http://172.16.0.1/api") is False
    assert is_safe_url("http://192.168.1.1") is False


@pytest.mark.asyncio
async def test_resolve_redirects_blocks_ssrf_redirect_hop():
    """Verify resolve_redirects aborts with SSRFError if a redirect target is an internal/loopback address."""
    def handler(request: httpx.Request) -> httpx.Response:
        # Initial request redirects to internal metadata IP
        return httpx.Response(302, headers={"Location": "http://169.254.169.254/secret"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(SSRFError):
            await resolve_redirects("https://safe-domain.com/redirect", client=client)


# ── 8. Mock DNS Resolver Network Isolation ────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_dns_resolver_prevents_network_and_respects_explicit_mock(monkeypatch):
    """Verify autouse mock DNS returns fixed public IP without network and explicit mocks override it."""
    # 1. Arbitrary test domain resolves to fixed public IP (93.184.216.34) without hitting network
    resolved_ips = await resolve_and_validate("arbitrary-never-registered-domain-999.example")
    assert len(resolved_ips) >= 1
    assert str(resolved_ips[0]) == "93.184.216.34"

    # 2. Test-level explicit mock overrides the default autouse fixture for private/blocked host testing
    def explicit_private_dns(host, port=0, family=0, type=0, proto=0, flags=0):
        if host == "custom-private-domain.test":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.1", port or 80))]
        return socket.getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", explicit_private_dns)

    with pytest.raises(SSRFError, match="prohibited IP address"):
        await resolve_and_validate("custom-private-domain.test")


# ── 9. URL Normalization and Deduplication ─────────────────────────────────────

def test_normalize_url():
    """Verify normalize_url strips fragments, tracking parameters, trailing slashes, and preserves path casing."""
    # Strip fragment
    assert normalize_url("https://example.com/page#overview") == "https://example.com/page"

    # Strip tracking parameters (utm_*, gclid, fbclid) and preserve other query parameters
    assert normalize_url("https://example.com/page?utm_source=google&q=search&gclid=123&fbclid=456&utm_campaign=spring") == "https://example.com/page?q=search"

    # Strip trailing slash on path
    assert normalize_url("https://example.com/docs/") == "https://example.com/docs"
    assert normalize_url("https://example.com/") == "https://example.com"

    # Do NOT lowercase path
    assert normalize_url("https://example.com/MyDocs/GettingStarted/?utm_medium=email") == "https://example.com/MyDocs/GettingStarted"


@pytest.mark.asyncio
async def test_discover_sources_normalization_dedupe():
    """Verify discovery dedupes on normalized URLs while preserving casing and original order."""
    chunks = [
        _make_web_chunk("https://example.com/Tutorial/?utm_source=twitter#section1", "Tutorial 1", "example.com"),
        _make_web_chunk("https://example.com/Tutorial", "Tutorial Duplicate", "example.com"),
        _make_web_chunk("https://example.com/Tutorial/?gclid=abc&fbclid=xyz", "Tutorial Tracker Duplicate", "example.com"),
        _make_web_chunk("https://example.com/OtherPage", "Other", "example.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sources = await discover_sources(
            query="FastAPI",
            max_results=5,
            provider=provider,
            http_client=client,
        )

    assert len(sources) == 2
    assert sources[0].url == "https://example.com/Tutorial/?utm_source=twitter#section1"
    assert sources[1].url == "https://example.com/OtherPage"


@pytest.mark.asyncio
async def test_discover_sources_403_kept_and_404_dropped():
    """Verify that a source whose final endpoint returns 403 is kept while a 404 source is dropped."""
    chunks = [
        _make_web_chunk("https://example.com/bot-blocked", "Blocked Page", "example.com"),
        _make_web_chunk("https://example.com/missing-404", "Missing Page", "example.com"),
        _make_web_chunk("https://example.com/accessible", "Good Page", "example.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/bot-blocked" in url_str:
            return httpx.Response(403, text="Forbidden")
        if "/missing-404" in url_str:
            return httpx.Response(404, text="Not Found")
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sources = await discover_sources(
            query="test status filtering",
            max_results=5,
            provider=provider,
            http_client=client,
        )

    assert len(sources) == 2
    assert sources[0].url == "https://example.com/bot-blocked"
    assert sources[1].url == "https://example.com/accessible"


@pytest.mark.asyncio
async def test_discover_sources_order_preserved_under_concurrency():
    """Verify concurrent chunk resolution preserves original chunk ordering even with disparate response times."""
    chunks = [
        _make_web_chunk("https://example.com/slow", "Slow Page", "example.com"),
        _make_web_chunk("https://example.com/fast1", "Fast 1", "example.com"),
        _make_web_chunk("https://example.com/fast2", "Fast 2", "example.com"),
        _make_web_chunk("https://example.com/medium", "Medium Page", "example.com"),
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    async def custom_handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/slow" in url_str:
            await asyncio.sleep(0.04)
        elif "/medium" in url_str:
            await asyncio.sleep(0.02)
        return httpx.Response(200, text="OK")

    transport = httpx.MockTransport(custom_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        sources = await discover_sources(
            query="concurrency ordering",
            max_results=5,
            provider=provider,
            http_client=client,
        )

    assert len(sources) == 4
    assert sources[0].url == "https://example.com/slow"
    assert sources[1].url == "https://example.com/fast1"
    assert sources[2].url == "https://example.com/fast2"
    assert sources[3].url == "https://example.com/medium"


@pytest.mark.asyncio
async def test_discover_sources_stop_resolving_redirects_at_max_results(monkeypatch):
    """Verify discover_sources resolves in ordered batches and stops once max_results unique URLs exist."""
    chunks = [
        _make_web_chunk(f"https://example.com/page{i}", f"Page {i}", "example.com")
        for i in range(10)
    ]
    mock_resp = _make_mock_response(chunks)

    provider = AsyncMock(spec=GeminiProvider)
    provider.generate_with_search.return_value = mock_resp

    resolved_uris: list[str] = []

    async def mock_resolve_redirects(url, max_redirects=3, client=None):
        resolved_uris.append(url)
        return url

    monkeypatch.setattr("app.research.discovery.resolve_redirects", mock_resolve_redirects)

    sources = await discover_sources(
        query="batch redirect resolution test",
        max_results=3,
        provider=provider,
    )

    assert len(sources) == 3
    # Batch size is DISCOVERY_CONCURRENCY (4). Only first batch of 4 is resolved,
    # remaining 6 chunks (page4 to page9) must not be resolved at all.
    assert len(resolved_uris) == 4
    assert "https://example.com/page0" in resolved_uris
    assert "https://example.com/page3" in resolved_uris
    assert "https://example.com/page4" not in resolved_uris
    assert "https://example.com/page9" not in resolved_uris
