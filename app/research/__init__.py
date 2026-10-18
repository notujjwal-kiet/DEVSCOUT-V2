"""
Research and source discovery package for DEVSCOUT.
"""
from app.research.discovery import discover_sources
from app.research.github import GitHubConnector
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
    "DiscoveredSource",
    "GitHubConnector",
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
    "reliability_score",
    "resolve_and_validate",
    "resolve_redirects",
    "score_document",
]

