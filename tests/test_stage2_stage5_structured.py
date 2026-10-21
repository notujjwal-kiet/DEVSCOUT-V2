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

    # Section 0 lost phantom citation and got marked unsupported, marker stripped from content
    assert repaired_exp.sections[0].citation_resource_ids == []
    assert repaired_exp.sections[0].support_status == "unsupported"
    assert NOT_BACKED_BY_SOURCES_MARKER not in repaired_exp.sections[0].content

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
    assert result.explanation.sections[0].support_status == "unsupported"
    assert NOT_BACKED_BY_SOURCES_MARKER not in result.explanation.sections[0].content


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


# ── Grounding, Excerpt, Budget, & Support Validation Tests ───────────────────

def test_grounded_section_passes_support_check():
    """Grounded section with high token overlap passes and is not marked weakly_supported."""
    resource = Resource(
        resource_id="res-redis-cluster",
        url="https://redis.io/topics/cluster-spec",
        type=ResourceType.DOCS,
        title="Redis Cluster Specification",
        content_hash="h123",
        reliability=1.0,
        freshness_score=0.9,
        excerpt=(
            "Redis Cluster provides a way to run a Redis installation where data is "
            "automatically sharded across multiple Redis nodes. Every cluster node has "
            "a unique identifier or node ID. Redis Cluster does not support multiple databases."
        ),
    )
    explanation = Explanation(
        topic="Redis Cluster",
        summary="Cluster summary",
        sections=[
            ExplanationSection(
                title="Cluster Architecture",
                content=(
                    "Redis Cluster allows running installations where data is automatically "
                    "sharded across multiple nodes using unique cluster node identifiers."
                ),
                citation_resource_ids=["res-redis-cluster"],
            )
        ],
        confidence_score=0.95,
    )

    report = validate_explanation(explanation, [resource])
    assert report.is_valid is True
    assert report.citations_valid is True
    assert explanation.sections[0].weakly_supported is False
    assert len(report.weakly_supported_sections) == 0


def test_cited_but_unrelated_content_flagged_weakly_supported():
    """Section citing a resource but containing unrelated content is flagged weakly_supported."""
    resource = Resource(
        resource_id="res-redis-cluster",
        url="https://redis.io/topics/cluster-spec",
        type=ResourceType.DOCS,
        title="Redis Cluster Specification",
        content_hash="h123",
        reliability=1.0,
        freshness_score=0.9,
        excerpt="Redis cluster hash slots and sharding across masters and replicas.",
    )
    explanation = Explanation(
        topic="Redis Cluster",
        summary="Cluster summary",
        sections=[
            ExplanationSection(
                title="Unrelated Topic",
                content=(
                    "Photosynthesis in green plants converts solar energy into chemical sugars "
                    "releasing oxygen as a byproduct."
                ),
                citation_resource_ids=["res-redis-cluster"],
            )
        ],
        confidence_score=0.9,
    )

    report = validate_explanation(explanation, [resource])
    # The citation is valid (the resource exists), so citations_valid is True
    assert report.citations_valid is True
    # But content has minimal/zero token overlap with excerpt -> flagged weakly_supported
    assert explanation.sections[0].weakly_supported is True
    assert report.weakly_supported_sections == ["Unrelated Topic"]
    # Marked clearly as a weak signal, not verification failure
    assert any("[Note: weak signal, not verification]" in issue for issue in report.issues)
    assert report.is_valid is True


def test_resource_excerpt_truncation_and_serialization_exclusion():
    from app.config import get_settings

    max_excerpt_chars = get_settings().max_excerpt_chars
    long_excerpt = "a" * (max_excerpt_chars + 1500)
    res = Resource(
        resource_id="res-long",
        url="https://example.com/long",
        type=ResourceType.DOCS,
        title="Long Excerpt Doc",
        content_hash="h999",
        reliability=1.0,
        freshness_score=0.9,
        excerpt=long_excerpt,
    )

    # Truncated in model
    assert len(res.excerpt) == max_excerpt_chars

    # Excluded when serialized to dict / JSON
    dumped = res.model_dump()
    assert "excerpt" not in dumped


@pytest.mark.asyncio
async def test_stage5_prompt_respects_excerpt_budget():
    """Stage 5 respects excerpt character budget across multiple resources."""
    r1 = Resource(
        resource_id="res-1",
        url="https://example.com/1",
        type=ResourceType.DOCS,
        title="Doc 1",
        content_hash="h1",
        reliability=1.0,
        freshness_score=1.0,
        excerpt="1234567890" * 50,  # 500 chars
    )
    r2 = Resource(
        resource_id="res-2",
        url="https://example.com/2",
        type=ResourceType.DOCS,
        title="Doc 2",
        content_hash="h2",
        reliability=0.9,
        freshness_score=0.9,
        excerpt="ABCDEFGHIJ" * 50,  # 500 chars
    )

    evidence = EvidenceReviewResult(
        goal_id="g1",
        is_sufficient=True,
        is_fresh=True,
        reviewed_resources=[r1, r2],
    )
    plan = LearningPlan(
        goal_id="g1",
        knowledge_gaps=[],
        learning_outcomes=[],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="Topic 1",
                description="Desc 1",
                target_outcome="Outcome 1",
            )
        ],
    )
    plan_result = IntentPlanResult(plan=plan)

    mock_provider = AsyncMock()
    mock_provider.generate_structured.return_value = ExplanationResult(
        explanation=Explanation(
            topic="Topic 1",
            summary="Summary",
            sections=[
                ExplanationSection(
                    title="Section 1",
                    content="Doc 1 content 1234567890",
                    citation_resource_ids=["res-1"],
                )
            ],
            confidence_score=0.95,
        ),
        concept_nodes=[],
        concept_edges=[],
    )

    # Budget of 300 chars: should not include full 1000 chars of excerpts
    await stage5_ai_analysis.run(
        evidence_review=evidence,
        plan_result=plan_result,
        provider=mock_provider,
        excerpt_budget=300,
        top_n_resources=2,
    )

    prompt = mock_provider.generate_structured.call_args.kwargs["prompt"]
    # Check that prompt contains excerpt instructions
    assert "Paraphrase concepts in your own words; DO NOT copy text verbatim" in prompt
    assert "[Not backed by sources]" in prompt

    # Extract all excerpts included in the prompt
    assert "1234567890" in prompt
    # Since total budget is 300, r1 takes ~300 chars, so r2 cannot have its full 500 chars
    assert prompt.count("ABCDEFGHIJ") < 50


@pytest.mark.asyncio
async def test_stage5_iterates_over_multiple_topics_from_milestones():
    """Stage 5 iterates over milestones/topics and includes top resources for each."""
    r_redis = Resource(
        resource_id="res-redis",
        url="https://redis.io/doc",
        type=ResourceType.DOCS,
        title="Redis Architecture",
        content_hash="h1",
        reliability=1.0,
        freshness_score=0.9,
        excerpt="Redis cluster hash slot allocation mechanism.",
    )
    r_fastapi = Resource(
        resource_id="res-fastapi",
        url="https://fastapi.tiangolo.com/doc",
        type=ResourceType.DOCS,
        title="FastAPI Integration",
        content_hash="h2",
        reliability=1.0,
        freshness_score=0.9,
        excerpt="FastAPI dependency injection and routing.",
    )

    evidence = EvidenceReviewResult(
        goal_id="g1",
        is_sufficient=True,
        is_fresh=True,
        reviewed_resources=[r_redis, r_fastapi],
    )
    plan = LearningPlan(
        goal_id="g1",
        knowledge_gaps=[],
        learning_outcomes=[],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="Redis Sharding",
                description="Cluster nodes",
                target_outcome="Sharded cluster",
            ),
            Milestone(
                title="FastAPI API Layer",
                description="Web service endpoints",
                target_outcome="Working API",
            ),
        ],
    )
    plan_result = IntentPlanResult(plan=plan)

    mock_provider = AsyncMock()
    mock_provider.generate_structured.return_value = ExplanationResult(
        explanation=Explanation(
            topic="Multiple Topics",
            summary="Summary",
            sections=[
                ExplanationSection(
                    title="Redis Sharding",
                    content="Redis cluster hash slot allocation.",
                    citation_resource_ids=["res-redis"],
                ),
                ExplanationSection(
                    title="FastAPI API Layer",
                    content="FastAPI dependency injection and routing.",
                    citation_resource_ids=["res-fastapi"],
                ),
            ],
            confidence_score=0.95,
        ),
        concept_nodes=[],
        concept_edges=[],
    )

    await stage5_ai_analysis.run(
        evidence_review=evidence,
        plan_result=plan_result,
        provider=mock_provider,
    )

    prompt = mock_provider.generate_structured.call_args.kwargs["prompt"]
    # Check that both topics appear in the prompt
    assert "Topic: Redis Sharding" in prompt
    assert "Topic: FastAPI API Layer" in prompt


@pytest.mark.asyncio
async def test_structured_llm_retries_only_on_validation_errors():
    """generate_structured_with_retry retries once on parse/validation errors, but never on RateLimitError or network error."""
    from app.llm.provider import RateLimitError, LLMProviderError
    from app.pipeline.structured_llm import generate_structured_with_retry
    from pydantic import BaseModel

    class DummyModel(BaseModel):
        ans: str

    # Case 1: RateLimitError -> Must NOT retry, raises immediately
    rate_limit_mock = AsyncMock()
    rate_limit_mock.generate_structured.side_effect = RateLimitError("429 Quota exhausted")

    with pytest.raises(RateLimitError):
        await generate_structured_with_retry(
            provider=rate_limit_mock,
            prompt="Test prompt",
            schema=DummyModel,
        )
    assert rate_limit_mock.generate_structured.await_count == 1

    # Case 2: Generic network error -> Must NOT retry
    net_mock = AsyncMock()
    net_mock.generate_structured.side_effect = LLMProviderError("Connection reset by peer")

    with pytest.raises(LLMProviderError, match="Connection reset"):
        await generate_structured_with_retry(
            provider=net_mock,
            prompt="Test prompt",
            schema=DummyModel,
        )
    assert net_mock.generate_structured.await_count == 1

    # Case 3: Validation / ValueError -> Retries once with correction prompt
    val_mock = AsyncMock()
    val_mock.generate_structured.side_effect = [
        ValueError("Schema validation failed: missing field"),
        DummyModel(ans="corrected"),
    ]

    result = await generate_structured_with_retry(
        provider=val_mock,
        prompt="Test prompt",
        schema=DummyModel,
    )
    assert result.ans == "corrected"
    assert val_mock.generate_structured.await_count == 2


