"""
Tests for app/pipeline/response_validator.py – Cross-cutting response validator.

Covers:
- valid response passes validation
- missing citation in an explanation claim
- nonexistent resource id cited
- code violating the gate (fenced code under EXPLAIN gate)
- code violating the gate (code exceeding 15 lines under SNIPPET gate)
- retry/repair hook resolving violations deterministically
"""
import pytest

from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ExperienceLevel,
    Explanation,
    ExplanationSection,
    LearnerProfile,
    LearningResponse,
    Resource,
    ResourceType,
    Snippet,
)
from app.pipeline.response_validator import repair_response, validate_response


def _make_valid_response() -> LearningResponse:
    res1 = Resource(
        url="https://docs.example.com",
        type=ResourceType.DOCS,
        title="Example Docs",
        content_hash="abc123hash",
        reliability=0.9,
        freshness_score=0.9,
        level_fit=ExperienceLevel.INTERMEDIATE,
    )
    profile = LearnerProfile(
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python", "FastAPI"],
    )
    explanation = Explanation(
        topic="Architecture",
        summary="Conceptual summary without code.",
        sections=[
            ExplanationSection(
                title="Foundations",
                content="Clear conceptual content with no raw code blocks.",
                citation_resource_ids=[res1.resource_id],
            )
        ],
        confidence_score=0.95,
    )
    gate_decision = CodeGateDecision(
        level=CodeGateLevel.EXPLAIN,
        reasons=["Default pedagogical policy: explain concepts."],
        violations=[],
    )
    return LearningResponse(
        learner_profile=profile,
        resources=[res1],
        explanation=explanation,
        code_gate_decision=gate_decision,
        confidence_score=0.95,
        quality_score=0.92,
    )


def test_valid_response_passes_validation():
    """A well-formed LearningResponse with valid citations and respected gate passes validation."""
    response = _make_valid_response()
    report = validate_response(response)
    assert report.is_valid is True
    assert report.schema_valid is True
    assert report.citations_valid is True
    assert report.code_gate_respected is True
    assert report.quality_metrics_present is True
    assert len(report.issues) == 0


def test_missing_citation_fails_validation():
    """An explanation section with no citations fails validation."""
    response = _make_valid_response()
    # Remove citation from the explanation section
    response.explanation.sections[0].citation_resource_ids = []

    report = validate_response(response)
    assert report.is_valid is False
    assert report.citations_valid is False
    assert any("has no citations" in issue for issue in report.issues)


def test_nonexistent_resource_id_fails_validation():
    """Citing a resource ID that is not present in response.resources fails validation."""
    response = _make_valid_response()
    # Cite an unknown resource ID
    response.explanation.sections[0].citation_resource_ids = ["nonexistent-uuid-1234"]

    report = validate_response(response)
    assert report.is_valid is False
    assert report.citations_valid is False
    assert any("nonexistent resource id" in issue for issue in report.issues)


def test_code_violating_gate_explain_with_fenced_block():
    """Fenced markdown code block present under EXPLAIN gate fails validation."""
    response = _make_valid_response()
    assert response.code_gate_decision.level == CodeGateLevel.EXPLAIN
    # Embed a fenced code block into explanation content
    response.explanation.sections[0].content = (
        "Here is some text:\n```python\ndef unauthorized():\n    return False\n```\nMore text."
    )

    report = validate_response(response)
    assert report.is_valid is False
    assert report.code_gate_respected is False
    assert any("fenced code block" in issue for issue in report.issues)


def test_code_violating_gate_snippet_over_15_lines():
    """Snippet with >15 lines under SNIPPET gate fails validation."""
    response = _make_valid_response()
    response.code_gate_decision.level = CodeGateLevel.SNIPPET

    # Put a 16-line fenced block in text
    code_16_lines = "\n".join([f"line_{i} = {i}" for i in range(16)])
    response.explanation.sections[0].content = f"```python\n{code_16_lines}\n```"

    report = validate_response(response)
    assert report.is_valid is False
    assert report.code_gate_respected is False
    assert any("exceeds 15 lines" in issue for issue in report.issues)


def test_repair_hook_fixes_unauthorized_code_and_citations():
    """Repair hook strips unauthorized code blocks and filters bad citations."""
    response = _make_valid_response()
    # Inject unauthorized code block and invalid citation
    response.explanation.sections[0].content = (
        "Concepts: ```python\nx = 1\n``` End."
    )
    response.explanation.sections[0].citation_resource_ids = [
        "fake-id",
        response.resources[0].resource_id,
    ]

    initial_report = validate_response(response)
    assert initial_report.is_valid is False

    repaired_resp, final_report = repair_response(response, initial_report)
    assert final_report.repaired is True
    assert final_report.is_valid is True
    assert final_report.code_gate_respected is True
    assert final_report.citations_valid is True
    assert "fake-id" not in repaired_resp.explanation.sections[0].citation_resource_ids
    assert "```python" not in repaired_resp.explanation.sections[0].content
