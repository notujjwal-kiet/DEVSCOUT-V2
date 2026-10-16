"""
app/research/models.py

Domain models and exceptions for DEVSCOUT research and source discovery.
Reuses Resource from app.models.pipeline.
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel

from app.models.pipeline import Resource, ResourceType


class ResearchError(Exception):
    """Raised when research or source connector encounters an error (network, rate limit, API)."""


class DiscoveredSource(BaseModel):
    """A source candidate discovered via search grounding prior to full review."""

    url: str
    title: str
    domain: str
    source_type_hint: ResourceType = ResourceType.BLOG


class ResearchDocument(BaseModel):
    """
    Raw material for Stage 4 evidence review.
    Scores (reliability, freshness, level fit) are filled in later during Stage 4.
    """

    resource: Resource
    text: str
    last_updated: datetime | None = None
    popularity: int | None = None

