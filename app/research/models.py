"""
app/research/models.py

Domain models and exceptions for DEVSCOUT research and source discovery.
Reuses Resource from app.models.pipeline.
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel

from app.models.pipeline import Resource


class ResearchError(Exception):
    """Raised when research or source connector encounters an error (network, rate limit, API)."""


class ResearchDocument(BaseModel):
    """
    Raw material for Stage 4 evidence review.
    Scores (reliability, freshness, level fit) are filled in later during Stage 4.
    """

    resource: Resource
    text: str
    last_updated: datetime | None = None
    popularity: int | None = None
