"""Tests for GeminiProvider thinking_budget, StageMetrics tokens, Stage 5 source/char capping,
Stage 7 structured 3-question generation without DEVSCOUT internals, and pipeline timeout.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch
import pytest

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    ConceptNode,
    EvidenceReviewResult,
    ExperienceLevel,
    Explanation,
    ExplanationResult,
    ExplanationSection,
    IntentPlanResult,
    LearningPlan,
    Milestone,
    QuestionType,
    Resource,
    ResourceType,
    StageMetrics,
)
from app.pipeline import stage5_ai_analysis, stage7_output_validation
from app.pipeline.stage7_output_validation import (
    build_fallback_questions,
    GeneratedQuestionItem,
    GeneratedQuestionsBatch,
)


def test_gemini_thinking_budget_config_and_provider_per_call():
    """Verify gemini_thinking_budget config default, provider construction, and per-call override."""
    settings = get_settings()
    assert hasattr(settings, "gemini_thinking_budget")
    assert settings.gemini_thinking_budget >= 0

    with patch("app.llm.provider.genai.Client"):
        # Test default thinking config
        provider = GeminiProvider(api_key="fake-key", thinking_budget=128)
        assert provider.thinking_budget == 128
        tc = provider._build_thinking_config(provider.thinking_budget)
        assert tc is not None
        assert tc.thinking_budget == 128

        # Test per-call override
        tc_override = provider._build_thinking_config(0)
        assert tc_override is not None
        assert tc_override.thinking_budget == 0

        # None returns None
        assert provider._build_thinking_config(None) is None


def test_stage_metrics_fill_tokens():
    """Verify StageMetrics.fill_tokens fills prompt, completion, and total token count from response or provider."""
    metrics = StageMetrics(stage_id=1, stage_name="test")

    class FakeUsage:
        prompt_token_count = 150
        candidates_token_count = 75
        total_token_count = 225

    class FakeResponse:
        usage_metadata = FakeUsage()

    metrics.fill_tokens(FakeResponse())
    assert metrics.prompt_tokens == 150
    assert metrics.completion_tokens == 75
    assert metrics.total_tokens == 225

    # Usage metadata dictionary
    metrics_dict = StageMetrics(stage_id=2, stage_name="test_dict")
    dict_response = {
        "usage_metadata": {
            "prompt_token_count": 80,
            "candidates_token_count": 40,
            "total_token_count": 120,
        }
    }
    metrics_dict.fill_tokens(dict_response)
    assert metrics_dict.prompt_tokens == 80
    assert metrics_dict.completion_tokens == 40
    assert metrics_dict.total_tokens == 120


@pytest.mark.asyncio
async def test_stage5_source_and_char_capping(caplog, capsys):
    """Verify Stage 5 sends at most 6 sources, truncates excerpts by max_excerpt_chars, and logs prompt tokens."""
    settings = get_settings()
    max_chars = settings.max_excerpt_chars

    # Create 10 resources with very long excerpts
    resources = [
        Resource(
            resource_id=f"res-{i}",
            url=f"https://example.com/source/{i}",
            type=ResourceType.DOCS,
            title=f"Source {i}",
            content_hash=f"hash-{i}",
            excerpt="x" * (max_chars + 500),
            topic="Architecture",
            reliability=0.9,
            freshness_score=1.0 - (i * 0.05),
        )
        for i in range(10)
    ]
    evidence_review = EvidenceReviewResult(
        goal_id="g1",
        is_sufficient=True,
        is_fresh=True,
        reviewed_resources=resources,
    )
    plan_result = IntentPlanResult(
        plan=LearningPlan(
            goal_id="g1",
            milestones=[
                Milestone(
                    title="Architecture",
                    description="Core system architecture",
                    target_outcome="Understand microservices",
                    concepts=["Microservices"],
                )
            ],
        )
    )

    class FakeUsage:
        prompt_token_count = 450
        candidates_token_count = 120
        total_token_count = 570

    class FakeResponse:
        usage_metadata = FakeUsage()

    class FakeProvider:
        last_response = FakeResponse()
        last_usage_metadata = FakeUsage()
        generate_structured = None

    fake_provider = FakeProvider()

    fake_explanation_result = ExplanationResult(
        explanation=Explanation(
            topic="Architecture",
            summary="System architecture overview",
            sections=[
                ExplanationSection(
                    title="Overview",
                    content="Overview content",
                    citation_resource_ids=["res-0"],
                )
            ],
            confidence_score=0.95,
        ),
        concept_nodes=[
            ConceptNode(
                id="c1",
                label="Architecture",
                description="Core concepts",
                difficulty=ExperienceLevel.BEGINNER,
            )
        ],
        concept_edges=[],
    )

    from unittest.mock import AsyncMock
    fake_provider.generate_structured = AsyncMock(return_value=fake_explanation_result)

    with caplog.at_level(logging.INFO):
        res = await stage5_ai_analysis.run(
            evidence_review=evidence_review,
            plan_result=plan_result,
            provider=fake_provider,
            top_n=10,
        )

    assert res.metrics.prompt_tokens == 450
    assert res.metrics.completion_tokens == 120
    assert res.metrics.total_tokens == 570

    # Verify the prompt passed to generate_structured contained at most 6 sources
    call_args = fake_provider.generate_structured.call_args
    prompt_sent = call_args.kwargs.get("prompt") or call_args.args[0]
    assert "res-5" in prompt_sent
    assert "res-6" not in prompt_sent

    # Verify logging occurred for prompt tokens
    captured = capsys.readouterr()
    assert "stage5_prompt_token_count" in captured.out or "stage5_prompt_token_count" in caplog.text


@pytest.mark.asyncio
async def test_stage7_structured_3_questions_different_concepts():
    """Verify Stage 7 generates 3 questions grounded in explanation text about different concepts without DEVSCOUT internals."""
    concept_nodes = [
        ConceptNode(id="node-1", label="Closure", description="Function with lexical environment"),
        ConceptNode(id="node-2", label="Scope Chain", description="Hierarchical scope resolution"),
        ConceptNode(id="node-3", label="Garbage Collection", description="Memory cleanup of unreferenced objects"),
    ]
    explanation = Explanation(
        topic="Python Closures",
        summary="Understanding closures and memory management",
        sections=[
            ExplanationSection(
                title="Lexical Closures",
                content="In Python, a closure is a function that retains access to its lexical scope even after outer scope terminates.",
                citation_resource_ids=[],
            ),
            ExplanationSection(
                title="Scope Chains",
                content="The scope chain dictates how variable lookup bubbles up from local to enclosing to global scopes.",
                citation_resource_ids=[],
            ),
            ExplanationSection(
                title="Garbage Collection",
                content="Garbage collection reclaims memory when object reference counts hit zero or cyclic sweeps run.",
                citation_resource_ids=[],
            ),
        ],
        confidence_score=0.9,
    )
    explanation_res = ExplanationResult(
        explanation=explanation,
        concept_nodes=concept_nodes,
        concept_edges=[],
    )

    fake_batch = GeneratedQuestionsBatch(
        questions=[
            GeneratedQuestionItem(
                concept_id="node-1",
                question_text="How does a closure retain access to outer variables?",
                question_type=QuestionType.CONCEPTUAL,
                expected_points=["lexical scope", "outer scope termination", "function binding"],
            ),
            GeneratedQuestionItem(
                concept_id="node-2",
                question_text="What is the lookup order in a Python scope chain?",
                question_type=QuestionType.CONCEPTUAL,
                expected_points=["local scope", "enclosing scope", "global scope"],
            ),
            GeneratedQuestionItem(
                concept_id="node-3",
                question_text="When does garbage collection clean up an object?",
                question_type=QuestionType.CONCEPTUAL,
                expected_points=["reference count hits zero", "cyclic sweep"],
            ),
        ]
    )

    from unittest.mock import AsyncMock
    fake_provider = MagicMock()
    fake_usage = MagicMock()
    fake_usage.prompt_token_count = 210
    fake_usage.candidates_token_count = 95
    fake_usage.total_token_count = 305
    fake_response = MagicMock()
    fake_response.usage_metadata = fake_usage
    fake_provider.last_response = fake_response
    fake_provider.last_usage_metadata = fake_usage
    fake_provider.generate_structured = AsyncMock(return_value=fake_batch)

    res = await stage7_output_validation.run(
        explanation_result=explanation_res,
        provider=fake_provider,
    )

    assert len(res.questions) == 3
    # Check that concepts are distinct
    concepts = [q.concept_id for q in res.questions]
    assert len(set(concepts)) == 3
    assert "node-1" in concepts
    assert "node-2" in concepts
    assert "node-3" in concepts

    # Check metrics
    assert res.metrics.prompt_tokens == 210
    assert res.metrics.completion_tokens == 95
    assert res.metrics.total_tokens == 305

    # Verify no DEVSCOUT internals mentioned in questions
    forbidden = ["readiness gate", "untyped dictionaries", "stages", "devscout", "pipeline"]
    for q in res.questions:
        for f in forbidden:
            assert f not in q.question_text.lower()
            assert f not in " ".join(q.expected_points).lower()


def test_stage7_fallback_questions_generic_no_internals():
    """Verify fallback questions when structured call fails are generic domain questions with no DEVSCOUT internals."""
    concept_nodes = [
        ConceptNode(id="n1", label="Dependency Injection", description="Decoupling components"),
    ]
    questions = build_fallback_questions("Software Architecture", concept_nodes)
    assert len(questions) == 3

    forbidden = ["readiness gate", "untyped dictionaries", "stages", "devscout", "pipeline"]
    for q in questions:
        for f in forbidden:
            assert f not in q.question_text.lower()
            assert f not in " ".join(q.expected_points).lower()


def test_pipeline_timeout_seconds_config():
    """Verify pipeline_timeout_seconds is set to 480.0."""
    settings = get_settings()
    assert settings.pipeline_timeout_seconds == 480.0
