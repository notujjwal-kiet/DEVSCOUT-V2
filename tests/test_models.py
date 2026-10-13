"""
Tests for app/models/pipeline.py – Pydantic domain models.
"""
import pytest
from pydantic import ValidationError

from app.models.pipeline import (
    DeveloperFeedback,
    DeveloperInput,
    EvidenceItem,
    Recommendation,
    StageMetrics,
)


def test_developer_input_requires_min_length():
    """question must be at least 10 characters."""
    with pytest.raises(ValidationError):
        DeveloperInput(question="short")


def test_developer_input_valid():
    d = DeveloperInput(question="PostgreSQL vs MongoDB for my SaaS project?")
    assert d.request_id != ""
    assert d.created_at is not None


def test_stage_metrics_stage_id_bounds():
    """stage_id must be between 1 and 9."""
    with pytest.raises(ValidationError):
        StageMetrics(stage_id=0, stage_name="bad")
    with pytest.raises(ValidationError):
        StageMetrics(stage_id=10, stage_name="bad")


def test_recommendation_confidence_bounds():
    """confidence must be in [0, 1]."""
    with pytest.raises(ValidationError):
        Recommendation(rank=1, option="X", rationale="Y", confidence=1.5)


def test_developer_feedback_rating_bounds():
    """rating must be in [1, 5]."""
    with pytest.raises(ValidationError):
        DeveloperFeedback(report_id="r1", request_id="q1", rating=0)
    with pytest.raises(ValidationError):
        DeveloperFeedback(report_id="r1", request_id="q1", rating=6)


def test_evidence_item_has_id():
    e = EvidenceItem(
        source_url="https://example.com",
        title="Test",
        snippet="Some text here",
    )
    assert e.evidence_id != ""
