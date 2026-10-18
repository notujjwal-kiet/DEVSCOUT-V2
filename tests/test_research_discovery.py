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
import socket
from unittest.mock import AsyncMock
import httpx
import pytest
from google.genai import types

from app.llm.provider import GeminiProvider
from app.models.pipeline import ResourceType
from app.research.discovery import (
    DOMAIN_TYPE_MAPPING,
    discover_sources,
    infer_source_type_hint,
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
