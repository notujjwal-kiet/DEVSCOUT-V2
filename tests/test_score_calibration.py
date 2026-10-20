"""
tests/test_score_calibration.py

Test Suite 3: Score Calibration
- Irrelevant sources plus unsupported claims -> confidence <= 0.35 and a critical warning present
- Property: adding a supported section never lowers confidence (monotonicity property)
- Scores never come from model-provided values (model-provided score is overridden)
Zero network calls: pure models and local response validator.
"""
from __future__ import annotations

from datetime import UTC, datetime
import pytest

from app.models.pipeline import (
    Citation,
    CodeGateDecision,
    CodeGateLevel,
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
from app.pipeline.response_validator import calibrate_confidence_score, validate_response


def _make_resource(
    res_id: str,
    title: str = "Test Source",
    url: str = "https://example.com/docs",
    reliability: float = 0.8,
    fetched_text: str = "FastAPI endpoints and request bodies are validated with Pydantic.",
    is_rejected: bool = False,
) -> Resource:
    return Resource(
        resource_id=res_id,
        url=url,
        type=ResourceType.DOCS,
        title=title,
        content_hash=f"hash_{res_id}",
        reliability=reliability,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
        fetched_text=fetched_text,
        is_rejected=is_rejected,
    )


def _make_response(
    resources: list[Resource],
    sections: list[ExplanationSection],
    model_confidence: float = 0.99,
) -> LearningResponse:
    plan = LearningPlan(
        goal_id="goal-calibration",
        knowledge_gaps=["Concurrency"],
        learning_outcomes=["Async APIs"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="API Setup",
                description="FastAPI initialization",
                target_outcome="Working app",
            )
        ],
    )
    explanation = Explanation(
        topic="Async FastAPI",
        summary="A pedagogical overview of async request handling.",
        sections=sections,
        confidence_score=model_confidence,
    )
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


def test_irrelevant_sources_and_unsupported_claims_low_confidence_and_critical_warning():
    """Irrelevant/rejected sources and unsupported claims yield confidence <= 0.35 and a critical warning."""
    # Source rejected during review
    bad_res = _make_resource(
        res_id="res-malware",
        title="Ransomware payload suite",
        url="https://github.com/bad/ransomware",
        reliability=0.0,
        fetched_text="Malicious payload",
        is_rejected=True,
    )
    # Section with unsupported claim / hallucinated quote
    bad_section = ExplanationSection(
        title="Payment Channels",
        content="Sends bitcoin ransom to anonymous wallets.",
        citations=[
            Citation(
                resource_id="res-malware",
                supporting_quote="Hallucinated quote not in text",
            )
        ],
    )

    resp = _make_response([bad_res], [bad_section], model_confidence=0.99)
    report = validate_response(resp)

    # Confidence must be calibrated to <= 0.35
    assert resp.explanation.confidence_score <= 0.35

    # Critical warning must be present (e.g. NO_RELIABLE_SOURCES)
    critical_warnings = [w for w in resp.warnings if w.severity == "critical"]
    assert len(critical_warnings) >= 1
    assert any(w.code == "NO_RELIABLE_SOURCES" for w in critical_warnings)

    # Warning for unsupported claims also present
    warning_codes = [w.code for w in resp.warnings]
    assert "UNSUPPORTED_CLAIMS" in warning_codes


def test_property_adding_supported_section_never_lowers_confidence():
    """Property: adding a supported section never lowers confidence (monotonically non-decreasing)."""
    res1 = _make_resource(
        res_id="res-1",
        reliability=0.9,
        fetched_text="FastAPI endpoints validate incoming request data using standard Python type annotations.",
    )
    res2 = _make_resource(
        res_id="res-2",
        reliability=0.85,
        fetched_text="Pydantic models perform runtime data parsing and schema validation.",
    )
    res_list = [res1, res2]

    # Test baseline with 1 supported, 1 unsupported section
    sec_supported_1 = ExplanationSection(
        title="Endpoints",
        content="FastAPI endpoints validate request data.",
        citations=[Citation(resource_id="res-1", supporting_quote="FastAPI endpoints validate incoming request data")],
        support_status="supported",
    )
    sec_unsupported = ExplanationSection(
        title="Unsupported Detail",
        content="An unverified claim without citations.",
        citations=[],
        support_status="unsupported",
    )

    # Base configuration: 1 supported, 1 unsupported
    base_sections = [sec_supported_1, sec_unsupported]
    base_score = calibrate_confidence_score(base_sections, res_list)

    # Add 1 additional supported section
    sec_supported_2 = ExplanationSection(
        title="Pydantic Validation",
        content="Pydantic parses data at runtime.",
        citations=[Citation(resource_id="res-2", supporting_quote="Pydantic models perform runtime data parsing")],
        support_status="supported",
    )
    plus_one_supported = [sec_supported_1, sec_unsupported, sec_supported_2]
    new_score = calibrate_confidence_score(plus_one_supported, res_list)

    assert new_score >= base_score, f"Adding supported section lowered score: {new_score} < {base_score}"

    # Add another supported section
    sec_supported_3 = ExplanationSection(
        title="Type Annotations",
        content="Standard Python type annotations are used throughout.",
        citations=[Citation(resource_id="res-1", supporting_quote="using standard Python type annotations")],
        support_status="supported",
    )
    plus_two_supported = [sec_supported_1, sec_unsupported, sec_supported_2, sec_supported_3]
    third_score = calibrate_confidence_score(plus_two_supported, res_list)

    assert third_score >= new_score, f"Adding another supported section lowered score: {third_score} < {new_score}"


def test_scores_never_come_from_model_provided_values():
    """Scores never come from model-provided values; model claim is overridden by deterministic calibration."""
    res = _make_resource(
        res_id="res-1",
        reliability=0.8,
        fetched_text="FastAPI endpoints validate incoming request data using standard Python type annotations.",
    )
    sec = ExplanationSection(
        title="Endpoints",
        content="FastAPI endpoints validate request data.",
        citations=[Citation(resource_id="res-1", supporting_quote="FastAPI endpoints validate incoming request data")],
        support_status="supported",
    )

    # Model provides unrealistic 1.0 confidence
    resp_high = _make_response([res], [sec], model_confidence=1.0)
    validate_response(resp_high)
    score_high = resp_high.explanation.confidence_score

    # Model provides unrealistic 0.05 confidence for the exact same valid evidence
    resp_low = _make_response([res], [sec], model_confidence=0.05)
    validate_response(resp_low)
    score_low = resp_low.explanation.confidence_score

    # Both must equal the deterministic calibrated value, completely ignoring the model-provided input
    assert score_high == score_low
    assert score_high != 1.0
    assert score_low != 0.05
