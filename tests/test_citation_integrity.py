"""
tests/test_citation_integrity.py

Test Suite 2: Citation Integrity & Verbatim Quote Anchoring
- A quote that exists verbatim passes
- An invented quote fails and is dropped (section becomes unsupported)
- A real quote from an irrelevant source cited for a FastAPI claim is flagged unsupported by relevance rules
- Citing a rejected resource ID fails and is dropped
Zero network calls: pure models and local response validator.
"""
from __future__ import annotations

from datetime import UTC, datetime
import pytest

from app.models.pipeline import (
    Citation,
    ExperienceLevel,
    Explanation,
    ExplanationSection,
    LearnerProfile,
    LearningPlan,
    LearningResponse,
    Milestone,
    Resource,
    ResourceType,
)
from app.pipeline.response_validator import validate_response


def _make_test_response(
    resources: list[Resource],
    sections: list[ExplanationSection],
) -> LearningResponse:
    plan = LearningPlan(
        goal_id="goal-test",
        knowledge_gaps=["Validation"],
        learning_outcomes=["Learn FastAPI"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="API Basics",
                description="FastAPI endpoints",
                target_outcome="Working API",
            )
        ],
    )
    explanation = Explanation(
        topic="FastAPI Web Framework",
        summary="A guide to FastAPI and request validation.",
        sections=sections,
        confidence_score=0.9,
    )
    from app.models.pipeline import CodeGateDecision, CodeGateLevel

    return LearningResponse(
        learner_profile=LearnerProfile(
            experience_level=ExperienceLevel.INTERMEDIATE,
            tech_stack=["Python", "FastAPI"],
        ),
        learning_plan=plan,
        explanation=explanation,
        resources=resources,
        guided_steps=[],
        questions=[],
        metrics=[],
        code_gate_decision=CodeGateDecision(
            level=CodeGateLevel.REQUIRE_CHECK,
            reasons=["Initial"],
            violations=[],
        ),
    )


def test_verbatim_quote_passes():
    """A citation carrying a quote that exists verbatim in fetched_text passes validation."""
    source_text = "FastAPI uses Pydantic models for request validation and serialization of data."
    res = Resource(
        resource_id="res-fastapi",
        url="https://fastapi.tiangolo.com/tutorial/",
        type=ResourceType.DOCS,
        title="FastAPI Tutorial",
        content_hash="hash1",
        reliability=1.0,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
        fetched_text=source_text,
    )

    section = ExplanationSection(
        title="Request Validation",
        content="FastAPI validates incoming request bodies using standard type hints.",
        citations=[
            Citation(
                resource_id="res-fastapi",
                supporting_quote="Pydantic models for request validation",
            )
        ],
    )

    resp = _make_test_response([res], [section])
    report = validate_response(resp)

    # Validated explanation section remains supported
    sec = resp.explanation.sections[0]
    assert sec.support_status == "supported"
    assert len(sec.citations) == 1
    assert sec.citations[0].supporting_quote == "Pydantic models for request validation"
    assert report.is_valid is True


def test_invented_quote_fails_and_is_dropped():
    """An invented / hallucinated quote not present in fetched_text is dropped, leaving section unsupported."""
    source_text = "FastAPI uses Pydantic models for request validation and serialization of data."
    res = Resource(
        resource_id="res-fastapi",
        url="https://fastapi.tiangolo.com/tutorial/",
        type=ResourceType.DOCS,
        title="FastAPI Tutorial",
        content_hash="hash1",
        reliability=1.0,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
        fetched_text=source_text,
    )

    section = ExplanationSection(
        title="Hallucinated Claim",
        content="FastAPI has built-in blockchain verification mechanisms.",
        citations=[
            Citation(
                resource_id="res-fastapi",
                supporting_quote="built-in blockchain verification mechanisms",
            )
        ],
    )

    resp = _make_test_response([res], [section])
    report = validate_response(resp)

    sec = resp.explanation.sections[0]
    # Dropped citation makes section unsupported
    assert len(sec.citations) == 0
    assert sec.support_status == "unsupported"
    assert report.is_valid is False
    assert any("unsupported" in err.lower() or "quote" in err.lower() or "citation" in err.lower() for err in report.issues)

    # Warning UNSUPPORTED_CLAIMS is attached to response
    warning_codes = [w.code for w in resp.warnings]
    assert "UNSUPPORTED_CLAIMS" in warning_codes


def test_real_quote_from_irrelevant_source_for_fastapi_claim_flagged():
    """A real quote from an irrelevant source (cookie recipe) cited for a FastAPI claim is flagged unsupported."""
    recipe_text = "Bake cookies at 350 degrees Fahrenheit for 12 minutes until golden brown."
    res_recipe = Resource(
        resource_id="res-recipe",
        url="https://recipes.example.com/chocolate-chip",
        type=ResourceType.BLOG,
        title="Classic Chocolate Chip Cookies",
        topic="Baking and desserts",
        content_hash="hash_recipe",
        reliability=0.5,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
        fetched_text=recipe_text,
    )

    section = ExplanationSection(
        title="FastAPI Request Pipeline",
        content="FastAPI handles async HTTP requests through Starlette.",
        citations=[
            Citation(
                resource_id="res-recipe",
                supporting_quote="Bake cookies at 350 degrees",
            )
        ],
    )

    resp = _make_test_response([res_recipe], [section])
    report = validate_response(resp)

    sec = resp.explanation.sections[0]
    # Citation dropped due to claim-source irrelevance
    assert len(sec.citations) == 0
    assert sec.support_status == "unsupported"
    assert any("irrelevant" in err.lower() or "unsupported" in err.lower() or "citation" in err.lower() for err in report.issues)


def test_citing_rejected_resource_id_fails():
    """Citing a resource that was rejected in evidence review fails and citation is dropped."""
    res_threat = Resource(
        resource_id="res-malware",
        url="https://github.com/bad/malware",
        type=ResourceType.REPO,
        title="Malware payload",
        content_hash="hash_threat",
        reliability=0.0,
        retrieved_at=datetime.now(UTC),
        fetched_text="Malicious payload encryption routines",
        is_rejected=True,
        rejection_reason="Untrusted security threat",
    )

    section = ExplanationSection(
        title="Security Architecture",
        content="Ensure data encryption in transit and at rest.",
        citations=[
            Citation(
                resource_id="res-malware",
                supporting_quote="Malicious payload encryption routines",
            )
        ],
    )

    resp = _make_test_response([res_threat], [section])
    report = validate_response(resp)

    sec = resp.explanation.sections[0]
    assert len(sec.citations) == 0
    assert sec.support_status == "unsupported"
    assert any("rejected" in err.lower() or "unsupported" in err.lower() or "citation" in err.lower() for err in report.issues)
