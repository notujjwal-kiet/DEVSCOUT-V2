"""
Tests for pipeline stage placeholders (stages 1-9) for DEVSCOUT Learning Companion.

Validates:
Stage 1 Developer Goal & Skill Context
Stage 2 Intent & Learning Plan
Stage 3 Research & Source Discovery
Stage 4 Evidence Review (including feedback loop 4->3 trigger)
Stage 5 Explanation & Concept Mapping
Stage 6 Guided Project Work
Stage 7 Understanding Check
Stage 8 Code Assistance Gate
Stage 9 Progress & Reflection
"""
import pytest

from app.models.pipeline import (
    CodeGateLevel,
    CodeGateResult,
    EvidenceReviewResult,
    ExperienceLevel,
    ExplanationResult,
    GoalContextResult,
    GuidedWorkResult,
    IntentPlanResult,
    ProgressReflectionResult,
    SourceDiscoveryResult,
    UnderstandingCheckResult,
)
from app.pipeline import (
    stage1_developer_input,
    stage2_input_processing,
    stage3_research,
    stage4_evidence_validation,
    stage5_ai_analysis,
    stage6_recommendation,
    stage7_output_validation,
    stage8_decision_report,
    stage9_developer_feedback,
)
import re
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    ConceptEdge,
    ConceptNode,
    Explanation,
    ExplanationSection,
    LearningPlan,
    Milestone,
    Resource,
    ResourceType,
)
from app.research.models import DiscoveredSource, ResearchDocument
from datetime import UTC, datetime


@pytest.fixture(autouse=True)
def mock_stage3_research_dependencies(monkeypatch):
    """Mock external discovery, github search, web fetch, and LLM structured generation in integration tests."""
    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://docs.python.org/3/library/asyncio.html",
                title="Python Asyncio Documentation",
                domain="docs.python.org",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://stackoverflow.com/questions/12345/rate-limiting",
                title="Rate Limiting Gotchas",
                domain="stackoverflow.com",
                source_type_hint=ResourceType.COMMUNITY,
            ),
        ]

    async def mock_search_repositories(self, query, limit=3):
        return [
            ResearchDocument(
                resource=Resource(
                    url="https://github.com/example/rate-limiter",
                    type=ResourceType.REPO,
                    title="example/rate-limiter",
                    content_hash="repo123",
                    retrieved_at=datetime.now(UTC),
                ),
                text="Example rate limiter README",
                last_updated=datetime.now(UTC),
                popularity=1500,
            )
        ]

    async def mock_fetch_page(url, resource_type, client=None):
        rtype = ResourceType(resource_type) if isinstance(resource_type, str) else resource_type
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=rtype,
                title=f"Title for {url}",
                content_hash="fetch123",
                retrieved_at=datetime.now(UTC),
            ),
            text=f"Fetched content for {url}",
            last_updated=datetime.now(UTC),
            popularity=None,
        )

    async def mock_generate_structured(self, prompt, schema, system_prompt=None, **kwargs):
        if schema == LearningPlan:
            return LearningPlan(
                goal_id="g1",
                knowledge_gaps=["Concurrency"],
                learning_outcomes=["Build limiter"],
                depth_level="intermediate",
                milestones=[
                    Milestone(
                        title="Setup",
                        description="Setup project",
                        target_outcome="Working baseline",
                        estimated_duration_minutes=30,
                    )
                ],
            )
        if schema == ExplanationResult:
            cites = re.findall(r"Resource ID: ([a-zA-Z0-9_-]+)", prompt)
            return ExplanationResult(
                explanation=Explanation(
                    topic="Distributed Rate Limiting",
                    summary="Understanding rate limiting algorithms.",
                    sections=[
                        ExplanationSection(
                            title="Token Bucket",
                            content="Token bucket algorithm description.",
                            citation_resource_ids=cites[:1] if cites else [],
                        )
                    ],
                    confidence_score=0.95,
                ),
                concept_nodes=[
                    ConceptNode(
                        id="c1",
                        label="Rate Limiter",
                        description="Token bucket",
                        difficulty=ExperienceLevel.BEGINNER,
                    )
                ],
                concept_edges=[
                    ConceptEdge(
                        source="c1",
                        target="c1",
                        relation="self",
                    )
                ],
            )
        from app.pipeline.stage7_output_validation import (
            GeneratedQuestionItem,
            GeneratedQuestionsBatch,
        )
        if schema == GeneratedQuestionsBatch:
            return GeneratedQuestionsBatch(
                questions=[
                    GeneratedQuestionItem(
                        concept_id="c1",
                        question_text="How does token replenishment rate interact with bucket capacity in token bucket algorithms?",
                        question_type=QuestionType.CONCEPTUAL,
                        expected_points=["Fixed replenishment rate", "Capacity limits bursts"],
                    ),
                    GeneratedQuestionItem(
                        concept_id="c2",
                        question_text="How do component boundaries and thread-safe operations protect token state under concurrency?",
                        question_type=QuestionType.EXPLAIN_BACK,
                        expected_points=["Thread-safe atomic updates", "Protects token state consistency"],
                    ),
                    GeneratedQuestionItem(
                        concept_id="c3",
                        question_text="What are the architectural trade-offs between token bucket and sliding window algorithms?",
                        question_type=QuestionType.EXERCISE,
                        expected_points=["Token bucket has O(1) memory", "Sliding window logs store request timestamps"],
                    ),
                ]
            )
        raise ValueError(f"Unexpected schema in mock: {schema}")

    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("GEMINI_API_KEY", "test-fake-key")
    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repositories)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)
    monkeypatch.setattr("app.llm.provider.GeminiProvider.generate_structured", mock_generate_structured)



@pytest.mark.asyncio
async def test_stage1_goal_context():
    """Stage 1 establishes learner profile and goal context."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python", "FastAPI", "Redis"],
        target_skills=["Distributed Systems", "Concurrency"],
    )
    assert isinstance(s1, GoalContextResult)
    assert s1.goal.title == "Build a Distributed Rate Limiter"
    assert s1.profile.experience_level == ExperienceLevel.INTERMEDIATE
    assert s1.metrics is not None
    assert s1.metrics.stage_id == 1


@pytest.mark.asyncio
async def test_stage2_learning_plan():
    """Stage 2 derives knowledge gaps, learning outcomes, and milestones."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python", "Redis"],
        target_skills=["Rate Limiting"],
    )
    s2 = await stage2_input_processing.run(s1)
    assert isinstance(s2, IntentPlanResult)
    assert s2.plan.goal_id == s1.goal.goal_id
    assert len(s2.plan.milestones) >= 1
    assert s2.metrics.stage_id == 2


@pytest.mark.asyncio
async def test_stage3_source_discovery():
    """Stage 3 discovers multi-perspective authoritative sources."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
    )
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    assert isinstance(s3, SourceDiscoveryResult)
    assert s3.goal_id == s1.goal.goal_id
    assert len(s3.resources) >= 2
    assert s3.metrics.stage_id == 3


@pytest.mark.asyncio
async def test_stage4_evidence_review_and_loop():
    """Stage 4 reviews evidence provenance and freshness, supporting 4->3 loop."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
    )
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    assert isinstance(s4, EvidenceReviewResult)
    assert s4.goal_id == s1.goal.goal_id
    assert s4.is_sufficient is True
    assert s4.metrics.stage_id == 4

    # Test loop trigger when evidence is marked insufficient or low freshness
    s3_empty = SourceDiscoveryResult(goal_id=s1.goal.goal_id, resources=[], loop_count=0)
    s4_needs_loop = await stage4_evidence_validation.run(s3_empty)
    assert s4_needs_loop.needs_more_evidence is True
    assert s4_needs_loop.is_sufficient is False


@pytest.mark.asyncio
async def test_stage5_concept_mapping():
    """Stage 5 creates explanations citing resources and builds concept graph."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
    )
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2)
    assert isinstance(s5, ExplanationResult)
    assert len(s5.explanation.sections) >= 1
    # Check that sections cite resources
    for sec in s5.explanation.sections:
        assert len(sec.citation_resource_ids) >= 1
    assert len(s5.concept_nodes) >= 1
    assert s5.metrics.stage_id == 5


@pytest.mark.asyncio
async def test_stage6_guided_work():
    """Stage 6 splits project into guided steps leaving implementation to learner."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
    )
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2)
    s6 = await stage6_recommendation.run(s5, s2)
    assert isinstance(s6, GuidedWorkResult)
    assert len(s6.steps) >= 1
    assert s6.steps[0].learner_task != ""
    assert s6.metrics.stage_id == 6


@pytest.mark.asyncio
async def test_stage7_understanding_check():
    """Stage 7 tests conceptual understanding and detects misconceptions."""
    s1 = await stage1_developer_input.run(
        goal_title="Build a Distributed Rate Limiter",
        goal_description="Understand token bucket and sliding window rate limiting in Python.",
    )
    s2 = await stage2_input_processing.run(s1)
    s3 = await stage3_research.run(s2)
    s4 = await stage4_evidence_validation.run(s3)
    s5 = await stage5_ai_analysis.run(s4, s2)
    s7 = await stage7_output_validation.run(
        s5, learner_response="Tokens are replenished at a constant rate and consumed on request."
    )
    assert isinstance(s7, UnderstandingCheckResult)
    assert len(s7.questions) >= 1
    assert len(s7.results) == 1
    assert s7.results[0].passed is True
    assert s7.metrics.stage_id == 7


@pytest.mark.asyncio
async def test_stage8_code_gate():
    """Stage 8 evaluates code gate policy."""
    s8 = await stage8_decision_report.run()
    assert isinstance(s8, CodeGateResult)
    assert s8.decision.level == CodeGateLevel.EXPLAIN
    assert s8.metrics.stage_id == 8


@pytest.mark.asyncio
async def test_stage9_progress_reflection():
    """Stage 9 persists learning history and reflections."""
    s9 = await stage9_developer_feedback.run(
        learner_id="learner-123",
        goal_id="goal-456",
        completed_topics=["Token Bucket Algorithm", "Redis sliding window"],
        reflection_notes="Great hands-on practice, now comfortable with distributed locking trade-offs.",
    )
    assert isinstance(s9, ProgressReflectionResult)
    assert s9.progress.learner_id == "learner-123"
    assert len(s9.progress.completed_topics) == 2
    assert s9.metrics.stage_id == 9
