"""
tests/test_code_gate_http.py

Test Suite 4: Code Assistance Gate HTTP Endpoints
- Unknown response_id -> 404
- Code-request before any completed job -> 404
- 1 of 3 answers -> no FULL
- 20 parallel requests cannot exceed the understanding attempt cap or skip the cooldown
- Unknown question IDs ignored
- Oversized attempt -> 413 or 422
- Provider failure during full-code generation -> no FULL, no partial code, generic error
Zero network calls: TestClient/AsyncClient, in-memory session store, fake provider.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import pytest
from starlette.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from app.api.auth import get_rate_limiter
from app.config import get_settings
from app.main import app
from app.models.pipeline import (
    AnswerSubmission,
    CodeGateDecision,
    CodeGateLevel,
    CodeRequest,
    ExperienceLevel,
    Explanation,
    ExplanationSection,
    LearnerProfile,
    LearningPlan,
    LearningResponse,
    Milestone,
    QuestionType,
    Resource,
    ResourceType,
    UnderstandingQuestion,
)
from app.pipeline.job_store import get_job_store
from app.pipeline.session_store import (
    create_session_from_response,
    get_session_store,
)


def _setup_test_session() -> tuple[str, list[UnderstandingQuestion]]:
    """Create and persist a complete test session with 3 questions in the session store."""
    store = get_session_store()
    questions = [
        UnderstandingQuestion(
            question_id=f"q_{i}",
            concept_id=f"c_{i}",
            question_text=f"Question text {i}",
            question_type=QuestionType.CONCEPTUAL,
            expected_points=[f"Key concept point {i}", f"Detail {i}"],
        )
        for i in range(1, 4)
    ]
    explanation = Explanation(
        topic="HTTP Gate Testing",
        summary="Testing code gate HTTP flows",
        sections=[
            ExplanationSection(
                title="Basics",
                content="Core concepts explanation",
                citation_resource_ids=["res-1"],
            )
        ],
        confidence_score=0.95,
    )
    plan = LearningPlan(
        goal_id="goal-http-test",
        knowledge_gaps=["Testing"],
        learning_outcomes=["Pass Gate"],
        depth_level="intermediate",
        milestones=[
            Milestone(
                title="Milestone 1",
                description="Gate testing",
                target_outcome="Working test",
            )
        ],
    )
    resource = Resource(
        resource_id="res-1",
        url="https://docs.python.org/3/",
        type=ResourceType.DOCS,
        title="Python Docs",
        content_hash="h1",
        reliability=1.0,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
    )
    lr = LearningResponse(
        learner_profile=LearnerProfile(
            experience_level=ExperienceLevel.INTERMEDIATE,
            tech_stack=["Python"],
        ),
        learning_plan=plan,
        explanation=explanation,
        resources=[resource],
        guided_steps=[],
        questions=questions,
        metrics=[],
        code_gate_decision=CodeGateDecision(
            level=CodeGateLevel.REQUIRE_CHECK,
            reasons=["Initial check"],
            violations=[],
        ),
    )
    session = create_session_from_response(lr)
    session.goal_title = "HTTP Code Gate Verification"
    session.goal_description = "Testing HTTP endpoints"
    if hasattr(store, "_store"):
        store._store[session.response_id] = session
    return lr.response_id, questions


@pytest.fixture(autouse=True)
async def clean_stores(monkeypatch):
    """Ensure in-memory stores and settings are reset before every test."""
    await get_session_store().clear()
    await get_job_store().clear()
    await get_rate_limiter().clear()
    cfg = get_settings()
    monkeypatch.setattr(cfg, "api_keys", "")
    monkeypatch.setattr(cfg, "rate_limit_capacity", 200.0)
    monkeypatch.setattr(cfg, "rate_limit_refill_rate", 100.0)
    monkeypatch.setattr(cfg, "max_understanding_attempts", 3)
    monkeypatch.setattr(cfg, "attempt_cooldown_seconds", 5.0)

    async def fake_code_gen(*args, **kwargs):
        from app.models.pipeline import FullCodeSolution
        return FullCodeSolution(
            code="def solution():\n    return 42\n",
            language="python",
            explanation="Test solution explanation",
            line_by_line_explanation=["Define function", "Return answer"],
        )

    monkeypatch.setattr(
        "app.pipeline.orchestrator.generate_structured_with_retry",
        fake_code_gen,
    )


def test_unknown_response_id_returns_404():
    """POST /api/code-request with unknown response_id returns 404 Not Found."""
    client = TestClient(app)
    response = client.post(
        "/api/code-request",
        json={"response_id": "nonexistent-response-id-12345", "wants_full_code": True},
    )
    assert response.status_code == 404
    assert "not found" in response.text.lower()


def test_code_request_before_completed_job_returns_404():
    """Requesting code before a job completes (or using a non-completed job id) returns 404."""
    client = TestClient(app)
    # Trigger background learn job
    learn_res = client.post(
        "/api/learn",
        json={
            "goal": "Build async HTTP server",
            "experience_level": "intermediate",
            "tech_stack": ["Python"],
        },
    )
    assert learn_res.status_code == 202
    job_id = learn_res.json()["job_id"]

    # Before job is completed, session does not exist under that ID -> 404
    code_res = client.post(
        "/api/code-request",
        json={"response_id": job_id, "wants_full_code": True},
    )
    assert code_res.status_code == 404


def test_one_of_three_answers_no_full():
    """Submitting only 1 of 3 required questions does not unlock FULL code."""
    resp_id, questions = _setup_test_session()
    client = TestClient(app)

    # Submit only 1 answer (need at least 2 and >= 60%)
    partial_answers = [
        {"question_id": questions[0].question_id, "answer": " ".join(questions[0].expected_points)}
    ]
    response = client.post(
        "/api/code-request",
        json={
            "response_id": resp_id,
            "wants_full_code": True,
            "answers": partial_answers,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["code_gate_decision"]["level"] != "FULL"
    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data.get("full_code") is None


@pytest.mark.asyncio
async def test_twenty_parallel_requests_cannot_exceed_cap_or_skip_cooldown(monkeypatch):
    """20 concurrent requests cannot exceed the understanding attempt cap or skip the cooldown."""
    from unittest.mock import AsyncMock
    # Explicitly disable rate limiter so this tests the attempt cap, not 429s
    monkeypatch.setattr("app.api.auth.RateLimiter.check", AsyncMock(return_value=True))

    resp_id, questions = _setup_test_session()
    answers = [
        {"question_id": q.question_id, "answer": "wrong answer"}
        for q in questions
    ]

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Launch 20 parallel attempts
        tasks = [
            client.post(
                "/api/code-request",
                json={
                    "response_id": resp_id,
                    "wants_full_code": True,
                    "answers": answers,
                },
            )
            for _ in range(20)
        ]
        responses = await asyncio.gather(*tasks)

    # All responses returned 200 without crashes
    assert all(r.status_code == 200 for r in responses)

    # Check that session attempt counter strictly did not exceed max_attempts (3)
    session = await get_session_store().get(resp_id)
    assert session is not None
    assert session.understanding_attempt_count <= 3

    # Many responses should be blocked by cooldown or max attempts
    rejection_reasons = [
        r.json()["code_gate_decision"]["reasons"]
        for r in responses
    ]
    has_cooldown_or_cap = any(
        any("cooldown" in reason.lower() or "maximum" in reason.lower() or "exceeded" in reason.lower() for reason in reasons)
        for reasons in rejection_reasons
    )
    assert has_cooldown_or_cap is True


def test_unknown_question_ids_ignored():
    """Answers with unknown question IDs are safely ignored and do not fulfill requirements."""
    resp_id, questions = _setup_test_session()
    client = TestClient(app)

    answers = [
        {"question_id": "bogus_unknown_id_999", "answer": "A totally valid answer to an unknown question."},
        {"question_id": questions[0].question_id, "answer": " ".join(questions[0].expected_points)},
    ]
    response = client.post(
        "/api/code-request",
        json={
            "response_id": resp_id,
            "wants_full_code": True,
            "answers": answers,
        },
    )
    assert response.status_code == 200
    data = response.json()
    # Only 1 valid question answered out of 3 -> no FULL
    assert data["code_gate_decision"]["level"] != "FULL"
    assert data.get("full_code") is None


def test_oversized_attempt_returns_413_or_422():
    """An attempt code payload that is oversized returns HTTP 413 or 422."""
    resp_id, _ = _setup_test_session()
    client = TestClient(app)

    oversized_code = "x = 1\n" * 15000  # > 50,000 characters
    response = client.post(
        "/api/code-request",
        json={
            "response_id": resp_id,
            "attempt_code": oversized_code,
            "wants_full_code": True,
        },
    )
    assert response.status_code in (413, 422)


def test_provider_failure_during_full_code_generation_no_full_no_partial_generic_error(monkeypatch):
    """Provider failure during full-code generation returns no FULL code, no partial code, and a generic error."""
    resp_id, questions = _setup_test_session()
    client = TestClient(app)

    passing_answers = [
        {"question_id": q.question_id, "answer": " ".join(q.expected_points)}
        for q in questions
    ]

    # Mock provider during full code generation to fail
    async def fake_failing_generate_structured(*args, **kwargs):
        raise RuntimeError("External LLM Provider timeout / unavailable")

    monkeypatch.setattr(
        "app.pipeline.orchestrator.generate_structured_with_retry",
        fake_failing_generate_structured,
    )

    response = client.post(
        "/api/code-request",
        json={
            "response_id": resp_id,
            "wants_full_code": True,
            "answers": passing_answers,
        },
    )

    # Handled via HTTP 500 or 200 with generic error and no code
    assert response.status_code in (200, 500)
    data = response.json()
    assert "full_code" not in data or data["full_code"] is None
    assert response.status_code != 200 or data.get("code_gate_decision", {}).get("level") != "FULL"
    assert "error" in response.text.lower() or "failed" in response.text.lower()
