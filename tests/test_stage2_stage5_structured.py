"""
tests/test_stage2_stage5_structured.py

Unit tests for:
- Stage 2 Intent & Learning Plan real call through GeminiProvider.generate_structured
- Stage 5 Explanation & Concept Mapping real call through GeminiProvider.generate_structured
- Citation validation & repair for Resource URLs/IDs from Stage 4
- Flagging and repairing uncited or phantom-cited claims with "[Not backed by sources]"
- Single retry with validation error reflection on invalid structured output
- Raising typed StructuredOutputError on persistent failure
- Full provider mocking with zero network calls
"""
from __future__ import annotations

import re
from unittest.mock import AsyncMock
import pytest
from pydantic import ValidationError

from app.models.pipeline import (
    ConceptEdge,
    ConceptNode,
    EvidenceReviewResult,
    ExperienceLevel,
    Explanation,
    ExplanationResult,
    ExplanationSection,
    GoalContextResult,
    IntentPlanResult,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    Milestone,
    Resource,
    ResourceType,
)
from app.pipeline import stage2_input_processing, stage5_ai_analysis
from app.pipeline.response_validator import (
    NOT_BACKED_BY_SOURCES_MARKER,
    repair_explanation,
    validate_explanation,
)
from app.pipeline.structured_llm import (
    StructuredOutputError,
    generate_structured_with_retry,
)


def _make_sample_goal_context() -> GoalContextResult:
    return GoalContextResult(
        profile=LearnerProfile(
            experience_level=ExperienceLevel.INTERMEDIATE,
            tech_stack=["Python", "FastAPI"],
            constraints=["Production latency < 50ms"],
        ),
        goal=LearningGoal(
            title="Distributed Caching System",
            description="Build and understand Redis cluster caching and invalidation.",
            target_skills=["Redis", "Distributed Caching"],
        ),
    )


def _make_sample_evidence_review() -> EvidenceReviewResult:
    r1 = Resource(
        resource_id="res-redis-docs",
        url="https://redis.io/docs/manual/clustering/",
        type=ResourceType.DOCS,
        title="Redis Clustering Guide",
        content_hash="hash1",
        reliability=1.0,
        freshness_score=0.9,
    )
    r2 = Resource(
        resource_id="res-fastapi-cache",
        url="https://github.com/example/fastapi-cache",
        type=ResourceType.REPO,
        title="FastAPI Cache Repository",
        content_hash="hash2",
        reliability=0.8,
        freshness_score=0.85,
    )
    return EvidenceReviewResult(
        goal_id="goal-123",
        is_sufficient=True,
        is_fresh=True,
        reviewed_resources=[r1, r2],
    )


def _make_sample_plan_result() -> IntentPlanResult:
    plan = LearningPlan(
        goal_id="goal-123",
        knowledge_gaps=["Cache Invalidation Strategies"],
        learning_outcomes=["Implement cache-aside pattern with TTL"],
        depth_level="advanced",
        milestones=[
            Milestone(
                title="Redis Setup",
                description="Cluster baseline",
                target_outcome="Working connection pool",
            )
        ],
    )
    return IntentPlanResult(plan=plan)


# ── Stage 2 Tests ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stage2_successful_structured_generation():
    """Stage 2 generates structured LearningPlan through GeminiProvider mock."""
    mock_provider = AsyncMock()
    mock_plan = LearningPlan(
        goal_id="temp_goal",
        knowledge_gaps=["Consistent Hashing"],
        learning_outcomes=["Understand ring topology"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="Ring Implementation",
                description="Implement virtual nodes",
                target_outcome="Balanced distribution",
                estimated_duration_minutes=45,
            )
        ],
    )
    mock_provider.generate_structured.return_value = mock_plan

    goal_context = _make_sample_goal_context()
    result = await stage2_input_processing.run(
        goal_context=goal_context,
        depth_level="deep_dive",
        provider=mock_provider,
    )

    assert isinstance(result, IntentPlanResult)
    assert result.plan.goal_id == goal_context.goal.goal_id
    assert result.plan.depth_level == "deep_dive"
    assert result.plan.knowledge_gaps == ["Consistent Hashing"]
    mock_provider.generate_structured.assert_awaited_once()

    # Ensure schema=LearningPlan was requested
    call_kwargs = mock_provider.generate_structured.call_args.kwargs
    assert call_kwargs["schema"] == LearningPlan


@pytest.mark.asyncio
async def test_stage2_retry_on_initial_validation_error():
    """Stage 2 retries once when initial output is invalid, appending error to prompt."""
    mock_provider = AsyncMock()
    valid_plan = LearningPlan(
        goal_id="temp_goal",
        knowledge_gaps=["Cache Invalidation"],
        learning_outcomes=["Solve stampede effect"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="Stampede Mitigation",
                description="Mutual exclusion lock",
                target_outcome="Working lock",
            )
        ],
    )
    # First attempt raises ValueError (simulating invalid JSON / validation failure)
    # Second attempt succeeds
    mock_provider.generate_structured.side_effect = [
        ValueError("Missing required field 'learning_outcomes'"),
        valid_plan,
    ]

    goal_context = _make_sample_goal_context()
    result = await stage2_input_processing.run(
        goal_context=goal_context,
        depth_level="intermediate",
        provider=mock_provider,
    )

    assert result.plan.goal_id == goal_context.goal.goal_id
    assert mock_provider.generate_structured.await_count == 2

    # Check that retry prompt included the validation error
    retry_prompt = mock_provider.generate_structured.call_args_list[1].kwargs["prompt"]
    assert "Missing required field 'learning_outcomes'" in retry_prompt
    assert "IMPORTANT: Your previous output failed validation" in retry_prompt


@pytest.mark.asyncio
async def test_stage2_raises_structured_output_error_after_retry_failure():
    """Stage 2 raises typed StructuredOutputError if retry also fails."""
    mock_provider = AsyncMock()
    mock_provider.generate_structured.side_effect = [
        ValueError("First failure"),
        ValueError("Second failure on retry"),
    ]

    goal_context = _make_sample_goal_context()
    with pytest.raises(StructuredOutputError) as exc_info:
        await stage2_input_processing.run(
            goal_context=goal_context,
            provider=mock_provider,
        )

    assert "LearningPlan" in str(exc_info.value)
    assert "Second failure on retry" in str(exc_info.value)
    assert mock_provider.generate_structured.await_count == 2


# ── Stage 5 Tests & Validator Integration ─────────────────────────────────────

@pytest.mark.asyncio
async def test_stage5_successful_generation_with_id_and_url_citations():
    """Stage 5 generates explanation citing Stage 4 resource IDs and URLs, normalized cleanly."""
    evidence = _make_sample_evidence_review()
    plan_result = _make_sample_plan_result()

    # Section 1 cites by ID ("res-redis-docs"), Section 2 cites by URL ("https://github.com/example/fastapi-cache")
    raw_result = ExplanationResult(
        explanation=Explanation(
            topic="Cache Invalidation",
            summary="Comprehensive explanation of caching.",
            sections=[
                ExplanationSection(
                    title="Redis Cluster Basics",
                    content="Redis cluster uses 16384 hash slots across nodes.",
                    citation_resource_ids=["res-redis-docs"],
                ),
                ExplanationSection(
                    title="Integration Pattern",
                    content="FastAPI cache integration decorators.",
                    citation_resource_ids=["https://github.com/example/fastapi-cache"],
                ),
            ],
            confidence_score=0.94,
        ),
        concept_nodes=[
            ConceptNode(
                id="n1",
                label="Hash Slots",
                description="Cluster partitioning",
                difficulty=ExperienceLevel.INTERMEDIATE,
            )
        ],
        concept_edges=[],
    )

    mock_provider = AsyncMock()
    mock_provider.generate_structured.return_value = raw_result

    result = await stage5_ai_analysis.run(
        evidence_review=evidence,
        plan_result=plan_result,
        provider=mock_provider,
        auto_repair=True,
    )

    assert isinstance(result, ExplanationResult)
    # Section 1 kept ID
    assert result.explanation.sections[0].citation_resource_ids == ["res-redis-docs"]
    # Section 2 URL was normalized to the corresponding resource_id
    assert result.explanation.sections[1].citation_resource_ids == ["res-fastapi-cache"]
    assert result.metrics is not None
    assert result.metrics.stage_id == 5


@pytest.mark.asyncio
async def test_stage5_validator_flags_phantom_citations_and_uncited():
    """Validator flags phantom citations and uncited sections."""
    evidence = _make_sample_evidence_review()

    explanation = Explanation(
        topic="Caching",
        summary="Summary",
        sections=[
            ExplanationSection(
                title="Phantom Section",
                content="Invented claims.",
                citation_resource_ids=["nonexistent-resource-id-999"],
            ),
            ExplanationSection(
                title="Uncited Section",
                content="Completely unreferenced claims without marker.",
                citation_resource_ids=[],
            ),
        ],
        confidence_score=0.9,
    )

    report = validate_explanation(explanation, evidence.reviewed_resources)
    assert report.is_valid is False
    assert report.citations_valid is False
    assert any("phantom" in issue for issue in report.issues)
    assert any("has no citations" in issue for issue in report.issues)


def test_validator_accepts_not_backed_by_sources_marker():
    """Validator accepts sections with no citations if marked with '[Not backed by sources]'."""
    evidence = _make_sample_evidence_review()

    explanation = Explanation(
        topic="Caching",
        summary="Summary",
        sections=[
            ExplanationSection(
                title="Speculative Future Design",
                content="Speculative pattern [Not backed by sources]",
                citation_resource_ids=[],
            )
        ],
        confidence_score=0.9,
    )

    report = validate_explanation(explanation, evidence.reviewed_resources)
    assert report.is_valid is True
    assert report.citations_valid is True


def test_repair_explanation_strips_phantom_and_marks_unbacked():
    """Repair hook strips phantom citations and marks claims with '[Not backed by sources]'."""
    evidence = _make_sample_evidence_review()

    explanation = Explanation(
        topic="Caching",
        summary="Summary",
        sections=[
            ExplanationSection(
                title="Phantom Section",
                content="This claim had only a fake citation.",
                citation_resource_ids=["https://phantom.example.com/unverified"],
            ),
            ExplanationSection(
                title="Partially Valid Section",
                content="This claim had both fake and real citations.",
                citation_resource_ids=["phantom-1", "res-redis-docs"],
            ),
        ],
        confidence_score=0.9,
    )

    initial_report = validate_explanation(explanation, evidence.reviewed_resources)
    assert initial_report.is_valid is False

    repaired_exp, repair_report = repair_explanation(
        explanation, evidence.reviewed_resources, initial_report
    )

    assert repair_report.repaired is True
    assert repair_report.is_valid is True

    # Section 0 lost phantom citation and got marked [Not backed by sources]
    assert repaired_exp.sections[0].citation_resource_ids == []
    assert NOT_BACKED_BY_SOURCES_MARKER in repaired_exp.sections[0].content

    # Section 1 kept only the valid resource id
    assert repaired_exp.sections[1].citation_resource_ids == ["res-redis-docs"]
    assert NOT_BACKED_BY_SOURCES_MARKER not in repaired_exp.sections[1].content


@pytest.mark.asyncio
async def test_stage5_auto_repair_in_pipeline():
    """Stage 5 automatically repairs phantom citations and marks unbacked claims."""
    evidence = _make_sample_evidence_review()
    plan_result = _make_sample_plan_result()

    raw_result = ExplanationResult(
        explanation=Explanation(
            topic="Distributed Caching",
            summary="Caching summary.",
            sections=[
                ExplanationSection(
                    title="Unverified Claim",
                    content="Claims from an unknown paper.",
                    citation_resource_ids=["fake-paper-uuid"],
                )
            ],
            confidence_score=0.95,
        ),
        concept_nodes=[],
        concept_edges=[],
    )

    mock_provider = AsyncMock()
    mock_provider.generate_structured.return_value = raw_result

    result = await stage5_ai_analysis.run(
        evidence_review=evidence,
        plan_result=plan_result,
        provider=mock_provider,
        auto_repair=True,
    )

    # Validated and repaired
    assert result.explanation.sections[0].citation_resource_ids == []
    assert NOT_BACKED_BY_SOURCES_MARKER in result.explanation.sections[0].content


@pytest.mark.asyncio
async def test_stage5_raises_structured_output_error_after_retry_failure():
    """Stage 5 raises typed StructuredOutputError when structured generation fails twice."""
    mock_provider = AsyncMock()
    mock_provider.generate_structured.side_effect = [
        ValueError("First failure in ExplanationResult generation"),
        ValueError("Second failure in ExplanationResult generation"),
    ]

    evidence = _make_sample_evidence_review()
    plan_result = _make_sample_plan_result()

    with pytest.raises(StructuredOutputError) as exc_info:
        await stage5_ai_analysis.run(
            evidence_review=evidence,
            plan_result=plan_result,
            provider=mock_provider,
        )

    assert "ExplanationResult" in str(exc_info.value)
    assert mock_provider.generate_structured.await_count == 2
