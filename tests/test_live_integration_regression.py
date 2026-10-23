"""
tests/test_live_integration_regression.py

Regression test suite covering the 5 live integration issue requirements:
1. Research query length under 256 chars for long goals, milestones, and tech stacks.
2. Grounded search routing: clarify that Gemini alone supports search grounding; Groq is inference-only.
3. Provider fallback across learning stages: Gemini exhausted -> Groq fallback without retrying exhausted Gemini.
4. Evidence integrity on empty search results: zero sources -> shortage, confidence <= 0.35, unsupported sections.
5. All providers unavailable: graceful degradation without unhandled 500 crashes.
"""
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.llm.provider import GeminiProvider, LLMProvider, RateLimitError
from app.llm.rate_gate import GeminiRateGate
from app.llm.registry import ModelEntry
from app.llm.router import RoutedProvider, can_run_grounded_search
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    EvidenceReviewResult,
    ExperienceLevel,
    Explanation,
    ExplanationResult,
    ExplanationSection,
    IntentPlanResult,
    LearningGoal,
    LearningPlan,
    LearnRequest,
    Milestone,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
)
from app.pipeline import orchestrator, stage3_research, stage5_ai_analysis
from app.research.github import GITHUB_QUERY_MAX_CHARS, GitHubConnector, build_github_query
from app.research.models import DiscoveredSource


# ── 1. Research Query Length Tests ───────────────────────────────────────────

def test_github_query_length_under_256_for_long_goal():
    """Ensure build_github_query produces queries strictly under 256 characters."""
    long_goal = (
        "Build a massively distributed, multi-region event-driven microservices architecture "
        "using Apache Kafka, Kubernetes cluster orchestration, GraphQL federation gateways, "
        "PostgreSQL partitioned databases, and Redis multi-tier caching with strict SLA rate limiting "
        "and zero-trust security architecture across hybrid-cloud AWS and GCP deployments."
    )
    long_milestone = (
        "Milestone 3: Implementing distributed sliding window counter rate limiter algorithm "
        "using Redis Lua scripts with atomic state transitions and token bucket fallbacks"
    )

    query1 = build_github_query(topic=long_goal, lang="python", stars_min=50)
    assert len(query1) <= GITHUB_QUERY_MAX_CHARS
    assert len(query1) < 200
    assert "language:python" in query1
    assert "stars:>50" in query1

    query2 = build_github_query(topic=long_milestone, lang="go")
    assert len(query2) <= GITHUB_QUERY_MAX_CHARS
    assert "language:go" in query2
    # Verify milestone prefix was stripped
    assert not query2.lower().startswith("milestone 3:")

    # Extremely long pathological input (> 1000 characters)
    extreme_topic = "deep learning " * 100
    query3 = build_github_query(topic=extreme_topic, lang="rust", stars_min=100)
    assert len(query3) <= GITHUB_QUERY_MAX_CHARS
    assert len(query3) > 0


@pytest.mark.asyncio
async def test_github_connector_enforces_256_limit():
    """GitHubConnector.search_repositories truncates query if raw string exceeds 256 chars."""
    captured_params: dict | None = None

    class MockResponse:
        status_code = 200
        def json(self):
            return {"items": []}

    class MockClient:
        async def get(self, url, params=None, headers=None):
            nonlocal captured_params
            captured_params = params
            return MockResponse()

    connector = GitHubConnector(client=MockClient())
    oversized_query = "a " * 200  # 400 characters
    await connector.search_repositories(query=oversized_query, limit=5)

    assert captured_params is not None
    q_sent = str(captured_params["q"])
    assert len(q_sent) <= GITHUB_QUERY_MAX_CHARS


# ── 2. Grounded Search Routing Tests ──────────────────────────────────────────

def test_grounded_search_routing_distinction():
    """Grounded web search is supported only by Gemini with quota; Groq is inference-only."""
    gemini_entry = ModelEntry(
        id="gemini-main",
        label="Gemini 2.5 Flash",
        provider="gemini",
        model="gemini-2.5-flash",
        priority=1,
    )
    groq_entry = ModelEntry(
        id="groq-openai/gpt-oss-120b",
        label="Groq GPT OSS 120B",
        provider="openai_compatible",
        model="openai/gpt-oss-120b",
        priority=2,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )

    mock_gate = MagicMock()
    mock_gate.has_budget.side_effect = lambda m_id: m_id == "gemini-main"

    with patch("app.llm.router.get_rate_gate", return_value=mock_gate):
        # Gemini with budget can run grounded search
        routed_gemini = RoutedProvider(primary_entry=gemini_entry)
        assert can_run_grounded_search(routed_gemini) is True

        # Groq cannot run grounded search (inference-only)
        routed_groq = RoutedProvider(primary_entry=groq_entry)
        assert can_run_grounded_search(routed_groq) is False

        # Gemini with exhausted budget cannot run grounded search
        mock_gate.has_budget.side_effect = lambda m_id: False
        assert can_run_grounded_search(routed_gemini) is False


# ── 3. Provider Fallback Across Stages (Gemini Quota -> Groq Fallback) ───────

@pytest.mark.asyncio
async def test_stage5_fallback_when_gemini_quota_exhausted():
    """When Gemini is exhausted, Stage 5 routes to Groq without calling exhausted Gemini."""
    gemini_entry = ModelEntry(
        id="gemini-main",
        label="Gemini 2.5 Flash",
        provider="gemini",
        model="gemini-2.5-flash",
        priority=1,
    )
    groq_entry = ModelEntry(
        id="groq-openai/gpt-oss-120b",
        label="Groq GPT OSS 120B",
        provider="openai_compatible",
        model="openai/gpt-oss-120b",
        priority=2,
        base_url="https://api.groq.com/openai/v1",
        api_key_env="GROQ_API_KEY",
    )

    mock_gate = MagicMock()
    # Gemini has NO budget; Groq HAS budget
    mock_gate.has_budget.side_effect = lambda m_id: m_id != "gemini-main"

    mock_groq_provider = AsyncMock(spec=LLMProvider)
    mock_groq_provider.model_id = "groq-openai/gpt-oss-120b"
    mock_groq_provider.generate_structured.return_value = ExplanationResult(
        explanation=Explanation(
            topic="Token Bucket Rate Limiting",
            summary="A robust rate limiting algorithm.",
            sections=[
                ExplanationSection(
                    title="Core Algorithm",
                    content="Tokens refill at a steady rate up to bucket capacity.",
                    citations=[],
                    support_status="supported",
                )
            ],
            confidence_score=0.9,
        ),
        concept_nodes=[],
        concept_edges=[],
    )

    with patch("app.llm.router.get_rate_gate", return_value=mock_gate), \
         patch.object(RoutedProvider, "_get_provider_instance", return_value=mock_groq_provider):

        routed = RoutedProvider(
            primary_entry=gemini_entry,
            fallback_entry=groq_entry,
            stage=5,
        )

        # Proactively bypassed exhausted primary
        assert routed.was_fallback is True
        assert routed.model_id == "groq-openai/gpt-oss-120b"

        evidence_review = EvidenceReviewResult(
            goal_id="goal-test",
            reviewed_resources=[],
            is_sufficient=False,
            is_fresh=True,
        )
        plan_result = IntentPlanResult(
            goal_id="goal-test",
            plan=LearningPlan(
                goal_id="goal-test",
                milestones=[
                    Milestone(
                        title="Token Bucket",
                        description="Rate limiter",
                        target_outcome="Understand rate limiter",
                    )
                ],
            ),
        )

        result = await stage5_ai_analysis.run(
            evidence_review=evidence_review,
            plan_result=plan_result,
            provider=routed,
        )

        assert result.explanation.topic == "Token Bucket Rate Limiting"
        assert result.metrics is not None
        assert result.metrics.model_id == "groq-openai/gpt-oss-120b"
        mock_groq_provider.generate_structured.assert_called_once()


# ── 4. Empty Search Results & Evidence Integrity ──────────────────────────────

@pytest.mark.asyncio
async def test_empty_search_results_maintains_evidence_integrity(monkeypatch):
    """When discovery returns 0 sources, shortage is declared, confidence <= 0.35, unsupported marked."""
    # Mock LLM provider to avoid live network calls
    mock_provider = AsyncMock(spec=LLMProvider)
    mock_provider.model_id = "gemini-main"

    async def fake_generate_structured(prompt, schema, system_prompt=None, **kwargs):
        if schema == LearningPlan:
            return LearningPlan(
                goal_id="goal-empty-search",
                knowledge_gaps=["Obscure Protocol Concept"],
                learning_outcomes=["Understand Obscure Protocol"],
                milestones=[
                    Milestone(
                        title="Obscure Topic",
                        description="No sources online",
                        target_outcome="Understand obscure topic",
                    )
                ],
            )
        elif schema == ExplanationResult:
            return ExplanationResult(
                explanation=Explanation(
                    topic="Obscure Topic",
                    summary="Explanation without sources.",
                    sections=[
                        ExplanationSection(
                            title="Unbacked Claims",
                            content="This section has no verified sources.",
                            citations=[],
                            support_status="supported",  # Should be changed to unsupported by pipeline
                        )
                    ],
                    confidence_score=0.8,
                ),
                concept_nodes=[],
                concept_edges=[],
            )
        return schema()

    mock_provider.generate_structured.side_effect = fake_generate_structured

    # Mock stage 3 to return 0 discovered resources
    async def mock_stage3_run(*args, **kwargs):
        return SourceDiscoveryResult(
            goal_id="goal-empty-search",
            discovered_resources=[],
            total_discovered=0,
            search_queries_used=["obscure topic"],
        )

    monkeypatch.setattr("app.pipeline.stage3_research.run", mock_stage3_run)

    req = LearnRequest(
        goal="Master an obscure unpublished protocol",
        experience_level=ExperienceLevel.INTERMEDIATE,
    )

    response = await orchestrator.run_learning_pipeline(req, provider=mock_provider)

    assert response.evidence_status == "insufficient"
    assert response.confidence_score <= 0.35
    assert len(response.resources) == 0

    # Verify NO_RELIABLE_SOURCES warning is present
    warning_codes = [w.code for w in response.warnings]
    assert "NO_RELIABLE_SOURCES" in warning_codes

    # Verify unsupported claims are marked as unsupported, not fabricated
    if response.explanation and response.explanation.sections:
        for section in response.explanation.sections:
            if not section.citations:
                assert section.support_status == "unsupported"


# ── 5. All Providers Unavailable (Graceful Degradation) ───────────────────────

@pytest.mark.asyncio
async def test_all_providers_unavailable_degrades_gracefully(monkeypatch):
    """When all providers are exhausted or fail, pipeline degrades gracefully without crashing."""
    async def mock_failing_provider_call(*args, **kwargs):
        raise RateLimitError("429: All configured AI providers are out of quota")

    failing_provider = AsyncMock(spec=LLMProvider)
    failing_provider.model_id = "gemini-main"
    failing_provider.generate.side_effect = mock_failing_provider_call
    failing_provider.generate_structured.side_effect = mock_failing_provider_call

    async def mock_stage3_run(*args, **kwargs):
        return SourceDiscoveryResult(
            goal_id="goal-fail",
            discovered_resources=[],
            total_discovered=0,
            search_queries_used=[],
        )

    monkeypatch.setattr("app.pipeline.stage3_research.run", mock_stage3_run)

    req = LearnRequest(
        goal="Learn Rust Memory Safety",
        experience_level=ExperienceLevel.BEGINNER,
    )

    response = await orchestrator.run_learning_pipeline(req, provider=failing_provider)

    # Pipeline completes and returns LearningResponse with fallback explanations
    assert response is not None
    assert response.explanation is not None
    assert response.confidence_score <= 0.35
    assert response.learning_plan is not None
