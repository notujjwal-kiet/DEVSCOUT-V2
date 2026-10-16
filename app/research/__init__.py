"""
Research and source discovery package for DEVSCOUT.
"""
from app.research.github import GitHubConnector
from app.research.models import ResearchDocument, ResearchError
from app.research.scoring import (
    content_changed,
    evidence_score,
    freshness_score,
    hash_content,
    normalize,
    reliability_score,
    score_document,
)

__all__ = [
    "GitHubConnector",
    "ResearchDocument",
    "ResearchError",
    "content_changed",
    "evidence_score",
    "freshness_score",
    "hash_content",
    "normalize",
    "reliability_score",
    "score_document",
]

