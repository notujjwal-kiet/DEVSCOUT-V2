"""
app/research/discovery.py

Search-grounded source discovery using Gemini with Google Search tool.
Extracts grounding_chunks from response metadata (not model prose),
resolves redirect links safely against SSRF, deduplicates by final URL,
and infers source type hints based on domain and path.
"""
from __future__ import annotations

import asyncio
import urllib.parse

import httpx
import structlog

from app.llm.provider import GeminiProvider
from app.models.pipeline import ResourceType
from app.research.models import DiscoveredSource
from app.research.web_fetch import resolve_redirects

logger = structlog.get_logger(__name__)

DISCOVERY_CONCURRENCY: int = 4

# ── Domain to ResourceType Mapping Configuration ──────────────────────────────

DOMAIN_TYPE_MAPPING: dict[str, ResourceType] = {
    "github.com": ResourceType.REPO,
    "stackoverflow.com": ResourceType.COMMUNITY,
    "reddit.com": ResourceType.COMMUNITY,
    "news.ycombinator.com": ResourceType.COMMUNITY,
    "arxiv.org": ResourceType.PAPER,
}

KNOWN_DOCS_DOMAINS: set[str] = {
    "docs.python.org",
    "fastapi.tiangolo.com",
    "docs.pydantic.dev",
    "docs.github.com",
    "docs.docker.com",
    "developer.mozilla.org",
    "kubernetes.io",
    "postgresql.org",
    "react.dev",
    "nodejs.org",
    "peps.python.org",
}


def normalize_url(url: str) -> str:
    """
    Normalize URL for deduplication:
    - Strips fragment (#...)
    - Strips tracking query parameters (utm_*, gclid, fbclid)
    - Strips trailing slash on path (e.g. /path/ -> /path, / -> empty)
    - Preserves path casing (does NOT lowercase path)
    - Lowercases scheme and netloc (domain:port)
    """
    if not url or not url.strip():
        return ""

    parsed = urllib.parse.urlsplit(url.strip())
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path.rstrip("/") if parsed.path else ""

    # Filter tracking query parameters
    query_params: list[tuple[str, str]] = []
    if parsed.query:
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True):
            k_lower = k.lower()
            if k_lower.startswith("utm_") or k_lower in ("gclid", "fbclid"):
                continue
            query_params.append((k, v))

    query_params.sort(key=lambda item: item[0])
    query = urllib.parse.urlencode(query_params)

    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def infer_source_type_hint(url: str, domain: str) -> ResourceType:
    """
    Infer the ResourceType hint from the domain and URL path:
    - github.com -> repo
    - stackoverflow.com, reddit.com, news.ycombinator.com -> community
    - known docs hosts or paths containing /docs -> docs
    - arxiv.org -> paper
    - else -> blog
    """
    clean_domain = domain.strip().lower()

    # 1. Direct domain mapping
    if clean_domain in DOMAIN_TYPE_MAPPING:
        return DOMAIN_TYPE_MAPPING[clean_domain]

    # Check subdomain matches (e.g. old.reddit.com -> community)
    for mapped_host, res_type in DOMAIN_TYPE_MAPPING.items():
        if clean_domain == mapped_host or clean_domain.endswith(f".{mapped_host}"):
            return res_type

    # 2. Known documentation domains
    if clean_domain in KNOWN_DOCS_DOMAINS:
        return ResourceType.DOCS
    for doc_host in KNOWN_DOCS_DOMAINS:
        if clean_domain == doc_host or clean_domain.endswith(f".{doc_host}"):
            return ResourceType.DOCS

    # 3. Path contains /docs
    if "/docs" in url.lower():
        return ResourceType.DOCS

    # 4. Default fallback: blog
    return ResourceType.BLOG


# ── Discover Sources ──────────────────────────────────────────────────────────

async def discover_sources(
    query: str,
    max_results: int = 8,
    provider: GeminiProvider | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> list[DiscoveredSource]:
    """
    Discover sources relevant to query using Gemini with Google Search grounding.
    - Plain text call with google_search tool (no JSON response_schema).
    - Extracts web URIs from grounding_metadata.grounding_chunks.
    - Resolves chunks concurrently using asyncio.gather with a semaphore of DISCOVERY_CONCURRENCY.
    - Preserves original chunk order.
    - Drops failed redirect resolutions.
    - Deduplicates on normalized URL (strips fragment, tracking params, trailing slash, keeps path casing).
    - Infers source_type_hint and caps results at max_results.
    - Returns [] and logs a warning on missing grounding metadata.
    """
    if not query or not query.strip():
        return []

    if provider is None:
        provider = GeminiProvider()

    search_prompt = f"Search documentation, repositories, articles, and discussions for: {query.strip()}"
    response = await provider.generate_with_search(search_prompt)

    # Extract grounding_metadata from first candidate or top-level response
    grounding_metadata = None
    if hasattr(response, "candidates") and response.candidates:
        grounding_metadata = getattr(response.candidates[0], "grounding_metadata", None)
    if grounding_metadata is None:
        grounding_metadata = getattr(response, "grounding_metadata", None)

    chunks = getattr(grounding_metadata, "grounding_chunks", None) if grounding_metadata else None
    if not chunks:
        logger.warning("No grounding metadata or chunks found in Gemini search response", query=query)
        return []

    valid_items: list[tuple[object, str, str, str]] = []
    for chunk in chunks:
        web = getattr(chunk, "web", None)
        if web is None:
            continue

        raw_uri = getattr(web, "uri", None)
        if not raw_uri:
            continue

        raw_title = getattr(web, "title", None) or ""
        fallback_domain = getattr(web, "domain", None) or ""
        valid_items.append((chunk, raw_uri, raw_title, fallback_domain))

    if not valid_items:
        return []

    # Resolve chunks concurrently with semaphore, preserving chunk order
    sem = asyncio.Semaphore(DISCOVERY_CONCURRENCY)

    async def _resolve_chunk(item: tuple[object, str, str, str]) -> tuple[tuple[object, str, str, str], str | None]:
        _, raw_uri, _, _ = item
        async with sem:
            try:
                final_url = await resolve_redirects(
                    url=raw_uri,
                    max_redirects=3,
                    client=http_client,
                )
                return (item, final_url)
            except Exception as exc:
                logger.warning(
                    "Failed to resolve search grounding URI, dropping source",
                    uri=raw_uri,
                    error=str(exc),
                )
                return (item, None)

    resolved_items = await asyncio.gather(*[_resolve_chunk(item) for item in valid_items])

    seen_normalized_urls: set[str] = set()
    discovered: list[DiscoveredSource] = []

    for item, final_url in resolved_items:
        if final_url is None:
            continue

        cleaned_final_url = final_url.strip()
        norm_url = normalize_url(cleaned_final_url)

        # Deduplicate on normalized URL while keeping order
        if norm_url in seen_normalized_urls:
            continue
        seen_normalized_urls.add(norm_url)

        _, _, raw_title, fallback_domain = item

        # Extract domain
        parsed = urllib.parse.urlparse(cleaned_final_url)
        domain = parsed.netloc.split(":")[0].lower() if parsed.netloc else fallback_domain.lower()

        type_hint = infer_source_type_hint(cleaned_final_url, domain)

        source = DiscoveredSource(
            url=cleaned_final_url,
            title=raw_title or cleaned_final_url,
            domain=domain,
            source_type_hint=type_hint,
        )
        discovered.append(source)

        if len(discovered) >= max_results:
            break

    return discovered
