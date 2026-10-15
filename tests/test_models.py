"""
Tests for app/models/pipeline.py – Strict Pydantic v2 domain models for DEVSCOUT Learning Companion.
"""
import pytest
from pydantic import ValidationError

from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ConceptEdge,
    ConceptNode,
    ExperienceLevel,
    Explanation,
    ExplanationSection,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    LearningResponse,
    Milestone,
    ProgressRecord,
    QuestionType,
    Resource,
    ResourceType,
    Snippet,
    StageMetrics,
    UnderstandingQuestion,
    UnderstandingResult,
)


def test_learning_goal_validation():
    """LearningGoal requires title and description with appropriate lengths."""
    with pytest.raises(ValidationError):
        LearningGoal(title="hi", description="too short")

    goal = LearningGoal(
        title="Build a Distributed Rate Limiter",
        description="Learn token bucket and sliding window rate limiting algorithms in Python.",
        target_skills=["Distributed Systems", "Redis", "Python"],
    )
    assert goal.goal_id != ""
    assert goal.created_at is not None


def test_learner_profile_defaults():
    """LearnerProfile assigns default UUID and Beginner level."""
    profile = LearnerProfile(
        tech_stack=["Python", "FastAPI"],
        constraints=["Local execution only"],
    )
    assert profile.learner_id != ""
    assert profile.experience_level == ExperienceLevel.BEGINNER


def test_stage_metrics_bounds():
    """stage_id must be within 1 and 9."""
    with pytest.raises(ValidationError):
        StageMetrics(stage_id=0, stage_name="invalid")
    with pytest.raises(ValidationError):
        StageMetrics(stage_id=10, stage_name="invalid")

    m = StageMetrics(stage_id=5, stage_name="Explanation & Concept Mapping")
    assert m.stage_id == 5


def test_resource_model():
    """Resource requires provenance, content hash, reliability, and freshness scores."""
    res = Resource(
        url="https://docs.python.org",
        type=ResourceType.DOCS,
        title="Python Documentation",
        content_hash="abc12345",
        reliability=0.95,
        freshness_score=0.90,
        level_fit=ExperienceLevel.BEGINNER,
    )
    assert res.resource_id != ""
    assert res.uncertainty_flag is False

    with pytest.raises(ValidationError):
        # Reliability out of bounds [0, 1]
        Resource(
            url="https://docs.python.org",
            type=ResourceType.DOCS,
            title="Docs",
            content_hash="abc",
            reliability=1.5,
            freshness_score=0.5,
            level_fit=ExperienceLevel.BEGINNER,
        )


def test_snippet_line_count_validation():
    """Snippet line_count must be <= 15."""
    with pytest.raises(ValidationError):
        Snippet(
            code="\n".join([f"line_{i} = {i}" for i in range(16)]),
            concept="Too long",
            line_count=16,
            explanation="Explanation text",
        )

    valid = Snippet(
        code="x = 1\ny = 2",
        concept="Assignment",
        line_count=2,
        explanation="Variables x and y are assigned.",
    )
    assert valid.line_count == 2


def test_concept_map_models():
    """ConceptNode and ConceptEdge represent graph elements."""
    node_a = ConceptNode(id="n1", label="Recursion", description="Function calling itself")
    node_b = ConceptNode(id="n2", label="Call Stack", description="LIFO call frames")
    edge = ConceptEdge(source="n1", target="n2", relation="uses")
    assert edge.source == "n1"
    assert edge.target == "n2"


def test_understanding_models():
    """UnderstandingQuestion and UnderstandingResult enforce scores and enums."""
    q = UnderstandingQuestion(
        concept_id="c1",
        question_text="What is the time complexity of binary search?",
        question_type=QuestionType.CONCEPTUAL,
    )
    assert q.question_type == QuestionType.CONCEPTUAL

    r = UnderstandingResult(
        question_id=q.question_id,
        passed=True,
        learner_response="O(log n) because search space halves each step.",
        feedback="Accurate reasoning.",
        score=1.0,
    )
    assert r.passed is True


def test_code_gate_decision_model():
    """CodeGateDecision requires level and reasons/violations lists."""
    dec = CodeGateDecision(
        level=CodeGateLevel.REQUIRE_CHECK,
        reasons=["Understanding check needed before code release."],
        violations=[],
    )
    assert dec.level == CodeGateLevel.REQUIRE_CHECK


def test_learning_response_composition():
    """LearningResponse composes all stage domain models without untyped dicts."""
    profile = LearnerProfile()
    dec = CodeGateDecision(level=CodeGateLevel.EXPLAIN, reasons=["Default explain"], violations=[])
    resp = LearningResponse(
        learner_profile=profile,
        code_gate_decision=dec,
        confidence_score=0.9,
        quality_score=0.95,
    )
    assert resp.response_id != ""
    assert resp.code_gate_decision.level == CodeGateLevel.EXPLAIN
