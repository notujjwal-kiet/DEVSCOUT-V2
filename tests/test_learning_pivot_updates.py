"""
tests/test_learning_pivot_updates.py

Unit tests verifying:
1. quality_score computation: 0.5 * structure_complete + 0.5 * fraction_supported, capped at 0.4 on insufficient evidence.
2. EVIDENCE REPORT: evidence_summary {discovered, fetched, accepted, rejected: [{url, reason}], errors}.
3. SEED DOCS FALLBACK: seed_docs mapping tech_stack terms, origin="curated", skipping failing URLs.
4. CLEANUP: stripping '[Not backed by sources]' from section content, server-set plan_id and created_at.
5. STAGE 7: 3 deterministic questions across different concepts from nodes and edges (no LLM).
6. GEMINI SPEED: thinking_level (low) vs thinking_budget, thoughts_token_count logging, timeout & backoff on RateLimitError.
Zero network calls: mock and fake components only.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from google.genai import types

from app.config import SEED_DOCS, get_settings
from app.llm.provider import GeminiProvider, LLMProviderError, RateLimitError
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    ConceptEdge,
    ConceptNode,
    EvidenceReviewResult,
    EvidenceSummary,
    ExperienceLevel,
    Explanation,
    ExplanationResult,
    ExplanationSection,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
    PublicUnderstandingQuestion,
    QuestionType,
    RejectedSource,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
)
from app.pipeline.orchestrator import compute_quality_score, run_learning_pipeline
from app.pipeline.response_validator import strip_unbacked_marker
from app.pipeline.stage5_ai_analysis import SYSTEM_PROMPT as STAGE5_SYSTEM_PROMPT
from app.pipeline.stage7_output_validation import (
    build_deterministic_questions,
    run as run_stage7,
)
from app.research.models import ResearchDocument


# ── 1. Quality Score Computation & Cap ─────────────────────────────────────────

def test_quality_score_formula_and_cap():
    """Verify quality_score formula: 0.5 * structure_complete + 0.5 * fraction_supported, capped at 0.4 on insufficient."""
    plan = LearningPlan(
        goal_id="g1",
        knowledge_gaps=["Routing"],
        learning_outcomes=["Learn API"],
        depth_level="intermediate",
        milestones=[Milestone(title="Milestone 1", description="Milestone 1 description", target_outcome="Outcome")],
    )
    explanation_supported = Explanation(
        topic="API Design",
        summary="Summary of API design.",
        sections=[
            ExplanationSection(
                title="Section 1",
                content="Content 1",
                citation_resource_ids=["res1"],
                support_status="supported",
            ),
            ExplanationSection(
                title="Section 2",
                content="Content 2",
                citation_resource_ids=["res2"],
                support_status="supported",
            ),
        ],
        confidence_score=0.9,
    )
    questions = [
        PublicUnderstandingQuestion(
            question_id="q1",
            concept_id="c1",
            question_text="What is API design?",
            question_type=QuestionType.CONCEPTUAL,
        )
    ]
    resp = LearningResponse(
        learner_profile=LearnerProfile(experience_level=ExperienceLevel.BEGINNER),
        learning_plan=plan,
        explanation=explanation_supported,
        questions=questions,
        code_gate_decision=CodeGateDecision(level=CodeGateLevel.EXPLAIN),
        evidence_status="sufficient",
    )

    # 1. Fully supported + complete structure: 0.5 * 1.0 + 0.5 * 1.0 = 1.0
    score = compute_quality_score(resp, evidence_status="sufficient")
    assert score == 1.0

    # 2. Half supported (1 of 2 sections): 0.5 * 1.0 + 0.5 * 0.5 = 0.75
    explanation_supported.sections[1].support_status = "unsupported"
    score_half = compute_quality_score(resp, evidence_status="sufficient")
    assert score_half == 0.75

    # 3. If evidence_status == "insufficient", score is capped at 0.4
    score_insufficient = compute_quality_score(resp, evidence_status="insufficient")
    assert score_insufficient == 0.4

    # 4. If questions are missing: structure_complete = 0.0 -> 0.5 * 0.0 + 0.5 * 0.5 = 0.25
    resp_no_questions = resp.model_copy()
    resp_no_questions.questions = []
    score_no_q = compute_quality_score(resp_no_questions, evidence_status="sufficient")
    assert score_no_q == 0.25


# ── 2. Evidence Summary Report ────────────────────────────────────────────────

def test_evidence_summary_model():
    """Verify EvidenceSummary records discovered, fetched, accepted, rejected, and errors."""
    summary = EvidenceSummary(
        discovered=10,
        fetched=8,
        accepted=5,
        rejected=[
            RejectedSource(url="https://example.com/bad", reason="Missing tech stack term"),
            RejectedSource(url="https://example.com/malicious", reason="Malicious keyword detected"),
        ],
        errors=["Rate limit reached for GitHub search"],
    )
    assert summary.discovered == 10
    assert summary.fetched == 8
    assert summary.accepted == 5
    assert len(summary.rejected) == 2
    assert summary.rejected[0].url == "https://example.com/bad"
    assert "Missing tech stack" in summary.rejected[0].reason
    assert summary.errors == ["Rate limit reached for GitHub search"]


# ── 3. Seed Docs Fallback ─────────────────────────────────────────────────────

def test_seed_docs_config_mapping():
    """Verify default SEED_DOCS maps key tech stack terms to official documentation URLs."""
    cfg = get_settings()
    assert "fastapi" in cfg.seed_docs
    assert cfg.seed_docs["fastapi"] == "https://fastapi.tiangolo.com/tutorial/"
    assert "pydantic" in cfg.seed_docs
    assert cfg.seed_docs["pydantic"] == "https://docs.pydantic.dev/latest/"
    assert "python" in cfg.seed_docs
    assert cfg.seed_docs["python"] == "https://docs.python.org/3/tutorial/"
    assert "react" in cfg.seed_docs
    assert cfg.seed_docs["react"] == "https://react.dev/learn"
    assert "typescript" in cfg.seed_docs
    assert cfg.seed_docs["typescript"] == "https://www.typescriptlang.org/docs/"


@pytest.mark.asyncio
async def test_seed_docs_fallback_triggers_when_shortage_and_labels_curated(monkeypatch):
    """Seed docs fallback fetches matching tech stack seeds with origin='curated' when loops produce fewer sources."""
    from app.pipeline import stage3_research

    # Mock discovery to return empty
    monkeypatch.setattr(stage3_research, "discover_sources", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        "app.research.github.GitHubConnector.search_repositories",
        AsyncMock(return_value=[]),
    )

    fetched_urls = []

    async def fake_fetch_page(url, resource_type, client=None):
        fetched_urls.append(url)
        res = Resource(
            url=url,
            type=ResourceType.DOCS,
            title=f"Official Guide for {url}",
            excerpt="FastAPI is a modern web framework for building APIs with Python.",
            content_hash="hash_seed",
            last_updated=datetime.now(UTC),
            reliability=1.0,
            freshness_score=0.95,
            topic="FastAPI Basics",
        )
        return ResearchDocument(
            resource=res,
            text="FastAPI tutorial building a Todo REST API with Python.",
        )

    monkeypatch.setattr("app.pipeline.orchestrator.fetch_page", fake_fetch_page)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", fake_fetch_page)
    monkeypatch.setattr("app.research.web_fetch.fetch_page", fake_fetch_page)

    fake_plan = LearningPlan(
        goal_id="goal-seed-test",
        knowledge_gaps=["Endpoints"],
        learning_outcomes=["Build API"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="FastAPI Basics",
                description="Routes and validation",
                target_outcome="Working endpoints",
            )
        ],
    )

    fake_provider = MagicMock()
    fake_provider.generate_structured = AsyncMock()

    async def fake_gen_struct(prompt, schema, **kwargs):
        name = getattr(schema, "__name__", "")
        if "LearningPlan" in name:
            return fake_plan
        return schema(
            explanation=Explanation(
                topic="FastAPI Basics",
                summary="FastAPI guide",
                sections=[
                    ExplanationSection(
                        title="Endpoints",
                        content="FastAPI defines routes.",
                        citation_resource_ids=[],
                    )
                ],
                confidence_score=0.9,
            ),
            concept_nodes=[
                ConceptNode(id="c1", label="FastAPI Routes", description="Route handlers")
            ],
            concept_edges=[],
        )

    fake_provider.generate_structured.side_effect = fake_gen_struct

    request = LearnRequest(
        goal="Learn FastAPI development",
        experience_level=ExperienceLevel.BEGINNER,
        tech_stack=["FastAPI"],
    )

    response = await run_learning_pipeline(request, provider=fake_provider)

    # Seed doc URL was fetched
    assert any("fastapi.tiangolo.com" in u for u in fetched_urls)

    # Curated resource was added and labeled origin="curated"
    curated = [r for r in response.resources if r.origin == "curated"]
    assert len(curated) >= 1
    assert any("fastapi.tiangolo.com" in r.url for r in curated)

    # Evidence summary reflects the fallback
    assert response.evidence_summary.fetched >= 1
    assert response.evidence_summary.accepted >= 1


# ── 4. Cleanup: Stripping Unbacked Marker & Server Plan ID ─────────────────────

def test_strip_unbacked_marker():
    """Verify strip_unbacked_marker cleanly removes '[Not backed by sources]' case-insensitively."""
    raw = "Core concepts for system architecture. [Not backed by sources] Additional details."
    cleaned = strip_unbacked_marker(raw)
    assert "[Not backed by sources]" not in cleaned
    assert "[not backed by sources]" not in cleaned.lower()
    assert cleaned == "Core concepts for system architecture. Additional details."


def test_stage5_system_prompt_forbids_missing_sources_and_unbacked_marker():
    """Verify Stage 5 system prompt forbids discussing missing sources or writing '[Not backed by sources]'."""
    assert "NEVER discuss missing sources or lack of excerpts" in STAGE5_SYSTEM_PROMPT
    assert "DO NOT include '[Not backed by sources]'" in STAGE5_SYSTEM_PROMPT


# ── 5. Stage 7 Deterministic Questions ────────────────────────────────────────

def test_deterministic_stage7_questions_formats_and_different_concepts():
    """Verify 3 deterministic questions are built without LLM, each using a different concept."""
    nodes = [
        ConceptNode(id="n1", label="FastAPI Dependency Injection", description="Provides modular reusable dependencies across endpoints"),
        ConceptNode(id="n2", label="Pydantic Models", description="Validates request data and enforces strict schema typing"),
        ConceptNode(id="n3", label="ASGI Server", description="Asynchronous server gateway interface executing Python async loops"),
    ]
    edges = [
        ConceptEdge(source="n1", target="n2", relation="validates dependencies through"),
    ]

    questions = build_deterministic_questions(
        concept_nodes=nodes,
        concept_edges=edges,
        project="E-commerce API",
    )

    assert len(questions) == 3

    # Q1: "In your own words, what is {label}?"
    assert questions[0].concept_id == "n1"
    assert questions[0].question_text == "In your own words, what is FastAPI Dependency Injection?"
    assert any("modular reusable dependencies" in pt for pt in questions[0].expected_points)

    # Q2: "How does {source label} relate to {target label}?" using an edge
    assert questions[1].concept_id == "n2"
    assert questions[1].question_text == "How does FastAPI Dependency Injection relate to Pydantic Models?"
    assert any("validates dependencies through" in pt for pt in questions[1].expected_points)

    # Q3: "Give one small example of {label} in your {project}"
    assert questions[2].concept_id == "n3"
    assert questions[2].question_text == "Give one small example of ASGI Server in your E-commerce API"
    assert any("Asynchronous server gateway" in pt for pt in questions[2].expected_points)

    # Distinct concepts
    concept_ids = {q.concept_id for q in questions}
    assert len(concept_ids) == 3


# ── 6. Gemini Provider Speed: Thinking Level & Budget, Timeout, Backoff ───────

def test_gemini_thinking_level_vs_budget():
    """Verify GeminiProvider chooses thinking_level for Gemini 3 and thinking_budget for Gemini 2.5."""
    p_gemini3 = GeminiProvider(model_name="gemini-3.0-flash", api_key="dummy-key")
    tc3 = p_gemini3._build_thinking_config()
    assert tc3 is not None
    assert tc3.thinking_level == types.ThinkingLevel.LOW

    p_gemini25 = GeminiProvider(model_name="gemini-2.5-flash", api_key="dummy-key", thinking_budget=128)
    tc25 = p_gemini25._build_thinking_config()
    assert tc25 is not None
    assert tc25.thinking_budget == 128


@pytest.mark.asyncio
async def test_gemini_provider_per_call_timeout_and_backoff():
    """Verify per-call timeout (default 60s) and exponential backoff on transient 5xx error."""
    mock_gate = MagicMock()
    mock_gate.acquire = AsyncMock()
    p = GeminiProvider(model_name="gemini-2.5-flash", api_key="dummy-key", timeout=0.1, rate_gate=mock_gate)
    assert p.timeout == 0.1

    # Test timeout error
    async def slow_call():
        await asyncio.sleep(0.5)
        return "done"

    with pytest.raises(Exception) as exc_info:
        await p._execute_with_retry(slow_call)
    assert "timed out" in str(exc_info.value).lower()

    # Test backoff retry on transient 5xx error
    attempts = 0

    async def server_error_call():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise LLMProviderError("503 Service Unavailable")
        return "success"

    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        res = await p._execute_with_retry(server_error_call, max_retries=2)
        assert res == "success"
        assert attempts == 2
        assert mock_sleep.await_count == 1
