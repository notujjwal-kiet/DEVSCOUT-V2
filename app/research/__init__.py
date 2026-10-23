"""
Research and source discovery package for DEVSCOUT.
"""
from app.research.discovery import (
    DISCOVERY_CONCURRENCY,
    discover_sources,
    normalize_url,
)
from app.research.github import GitHubConnector, build_github_query
from app.research.models import DiscoveredSource, ResearchDocument, ResearchError
from app.research.scoring import (
    content_changed,
    evidence_score,
    freshness_score,
    hash_content,
    normalize,
    reliability_score,
    score_document,
)
from app.research.web_fetch import (
    SSRFError,
    WebFetchError,
    fetch_page,
    is_safe_url,
    resolve_and_validate,
    resolve_redirects,
)

__all__ = [
    "DISCOVERY_CONCURRENCY",
    "DiscoveredSource",
    "GitHubConnector",
    "build_github_query",
    "ResearchDocument",
    "ResearchError",
    "SSRFError",
    "WebFetchError",
    "content_changed",
    "discover_sources",
    "evidence_score",
    "fetch_page",
    "freshness_score",
    "hash_content",
    "is_safe_url",
    "normalize",
    "normalize_url",
    "reliability_score",
    "resolve_and_validate",
    "resolve_redirects",
    "score_document",
]

