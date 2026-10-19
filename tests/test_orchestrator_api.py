"""
tests/test_orchestrator_api.py

End-to-end integration tests for:
1. Full happy path for POST /api/learn -> LearningResponse
2. Stage 4 evidence review triggering retry loop back to Stage 3 once
3. Trivial code attempt does not unlock FULL code (returns REQUIRE_CHECK)
4. Passing answers unlock FULL code with line-by-line explanation
5. Failing answers return REQUIRE_CHECK with questions
6. Unknown response_id returns 404 Not Found
7. Code gate violation blocks unauthorized code response
Zero network calls: fake provider, mocked research discovery, and local session store.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models.pipeline import (
    AnswerSubmission,
    CodeGateDecision,
    CodeGateLevel,
    CodeRequest,
    ConceptNode,
    EvidenceReviewResult,
    ExperienceLevel,
    Explanation,
    ExplanationSection,
    FullCodeSolution,
    LearnerProfile,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
    Resource,
    ResourceType,
    Snippet,
    UnderstandingQuestion,
    UnderstandingResult,
)
from app.pipeline.orchestrator import (
    handle_code_request,
    run_learning_pipeline,
)
from app.pipeline.response_validator import validate_response
from app.pipeline.session_store import (
    InMemorySessionStore,
    SessionData,
    create_session_from_response,
    get_session_store,
)
from app.policy.code_gate import (
    LearnerReadiness,
    evaluate_code_gate,
)
from app.research.models import DiscoveredSource, ResearchDocument


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Set up mocks for LLM provider and research connectors to eliminate network calls."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-mock-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Clear in-memory session store for clean test isolation
    store = get_session_store()
    if isinstance(store, InMemorySessionStore):
        store._store.clear()

    async def fake_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://docs.python.org/3/library/asyncio.html",
                title="Python Asyncio Documentation",
                domain="docs.python.org",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://github.com/example/rate-limiter",
                title="Rate Limiter Implementation",
                domain="github.com",
                source_type_hint=ResourceType.REPO,
            ),
        ]

    async def fake_search_repositories(self, query, limit=3):
        return []

    async def fake_fetch_page(url, resource_type, client=None):
        rtype = ResourceType(resource_type) if isinstance(resource_type, str) else resource_type
        return ResearchDocument(
            resource=Resource(
                resource_id="res-docs-1",
                url=url,
                type=rtype,
                title=f"Authoritative Guide for {url}",
                content_hash="fetchhash123",
                reliability=1.0,
                freshness_score=0.9,
                retrieved_at=datetime.now(UTC),
                excerpt="Tokens are added at a fixed replenishment rate up to bucket capacity.",
            ),
            text="Tokens are added at a fixed replenishment rate up to bucket capacity.",
        )

    async def fake_generate_structured(self, prompt, schema, system_prompt=None):
        if schema == LearningPlan:
            return LearningPlan(
                goal_id="goal-test-123",
                knowledge_gaps=["Token Bucket Algorithm", "Sliding Window"],
                learning_outcomes=["Implement token bucket in Python"],
                depth_level="intermediate",
                milestones=[
                    Milestone(
                        title="Token Bucket Algorithm",
                        description="Implement token bucket algorithm in Python",
                        target_outcome="Working bucket class",
                    )
                ],
            )
        from app.models.pipeline import ExplanationResult

        if schema == ExplanationResult:
            return ExplanationResult(
                explanation=Explanation(
                    topic="Token Bucket Algorithm",
                    summary="Comprehensive pedagogical explanation of token bucket algorithm.",
                    sections=[
                        ExplanationSection(
                            title="Core Mechanics",
                            content="Tokens are added at a fixed replenishment rate up to bucket capacity.",
                            citation_resource_ids=["res-docs-1"],
                        )
                    ],
                    confidence_score=0.95,
                ),
                concept_nodes=[
                    ConceptNode(
                        id="c1",
                        label="Token Bucket",
                        description="Rate limiter bucket algorithm",
                        difficulty=ExperienceLevel.INTERMEDIATE,
                    )
                ],
                concept_edges=[],
            )
        if schema == FullCodeSolution:
            return FullCodeSolution(
                code=(
                    "class TokenBucket:\n"
                    "    def __init__(self, capacity: int, fill_rate: float):\n"
                    "        self.capacity = capacity\n"
                    "        self.fill_rate = fill_rate\n"
                    "        self.tokens = capacity\n"
                ),
                language="python",
                explanation="Production-grade token bucket implementation.",
                line_by_line_explanation=[
                    "Define TokenBucket class",
                    "Store maximum capacity and replenishment fill rate",
                    "Initialize bucket tokens to full capacity",
                ],
            )
        raise ValueError(f"Unexpected schema in fake_generate_structured: {schema}")

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", fake_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", fake_search_repositories)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", fake_fetch_page)
    monkeypatch.setattr("app.llm.provider.GeminiProvider.generate_structured", fake_generate_structured)


# ── Test 1: Full Happy Path ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_happy_path_post_api_learn():
    """POST /api/learn executes end-to-end pipeline, returning complete LearningResponse."""
    payload = {
        "goal": "Build a Distributed Rate Limiter with Redis",
        "project": "API Gateway",
        "experience_level": "intermediate",
        "tech_stack": ["Python", "Redis", "FastAPI"],
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/learn", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "response_id" in data
    assert data["learning_plan"] is not None
    assert len(data["learning_plan"]["milestones"]) >= 1
    assert len(data["resources"]) >= 1
    assert data["explanation"] is not None
    assert len(data["explanation"]["sections"]) >= 1

    # 2-3 understanding questions with expected points
    assert len(data["questions"]) in (2, 3)
    for q in data["questions"]:
        assert len(q["expected_points"]) >= 1

    # Default code gate is EXPLAIN
    assert data["code_gate_decision"]["level"] == "EXPLAIN"

    # Telemetry metrics present for executed stages
    assert len(data["metrics"]) >= 5

    # Saved in session store
    store = get_session_store()
    session = await store.get(data["response_id"])
    assert session is not None
    assert session.response_id == data["response_id"]


# ── Test 2: Evidence Loop Triggered Once ───────────────────────────────────────

@pytest.mark.asyncio
async def test_evidence_loop_triggered_once(monkeypatch):
    """Stage 4 triggers retry loop back to Stage 3 with seen URLs when evidence is initially insufficient."""
    from app.pipeline import stage3_research, stage4_evidence_validation

    call_counts = {"stage3": 0, "stage4": 0}
    stage3_received_loops: list[int] = []
    stage3_received_excludes: list[list[str]] = []

    real_stage3_run = stage3_research.run
    real_stage4_run = stage4_evidence_validation.run


    async def mock_stage3_tracking(*args, **kwargs):
        call_counts["stage3"] += 1
        stage3_received_loops.append(kwargs.get("loop_count", 0))
        stage3_received_excludes.append(list(kwargs.get("exclude_urls", [])))
        return await real_stage3_run(*args, **kwargs)

    async def mock_stage4_initial_insufficient(*args, **kwargs):
        call_counts["stage4"] += 1
        res = await real_stage4_run(*args, **kwargs)
        if call_counts["stage4"] == 1:
            # First attempt: insufficient, requests more evidence
            res.is_sufficient = False
            res.needs_more_evidence = True
            res.seen_urls = ["https://docs.python.org/3/library/asyncio.html"]
        else:
            # Second attempt: sufficient
            res.is_sufficient = True
            res.needs_more_evidence = False
        return res

    monkeypatch.setattr("app.pipeline.stage3_research.run", mock_stage3_tracking)
    monkeypatch.setattr("app.pipeline.stage4_evidence_validation.run", mock_stage4_initial_insufficient)

    request = LearnRequest(
        goal="Distributed Rate Limiter",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )

    result = await run_learning_pipeline(request)

    # Loop ran: Stage 3 called twice, Stage 4 called twice
    assert call_counts["stage3"] == 2
    assert call_counts["stage4"] == 2

    # Loop counts passed to Stage 3 were 0, then 1
    assert stage3_received_loops == [0, 1]

    # Exclude URLs in second call contained URL seen in first loop
    assert "https://docs.python.org/3/library/asyncio.html" in stage3_received_excludes[1]
    assert result.explanation is not None


# ── Test 3: Trivial Attempt Does Not Unlock FULL ───────────────────────────────

@pytest.mark.asyncio
async def test_trivial_attempt_does_not_unlock_full():
    """Trivial attempt (< 10 real code lines or no structure) does not unlock FULL code."""
    # Seed a learning session
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id

    # Submit trivial code attempt (<10 lines, only assignments)
    payload = {
        "response_id": resp_id,
        "wants_full_code": True,
        "attempt_code": "a = 1\nb = 2\nc = 3\n",
        "attempt_language": "python",
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data["full_code"] is None
    assert len(data["questions"]) >= 1
    assert any("readiness criteria not met" in r.lower() for r in data["code_gate_decision"]["reasons"])


# ── Test 4: Passing Answers Unlock FULL ────────────────────────────────────────

@pytest.mark.asyncio
async def test_passing_answers_unlock_full():
    """Submitting answers covering question expected_points unlocks FULL code release."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions

    # Provide answers with high token overlap with expected points
    q1 = questions[0]
    q2 = questions[1]

    answers = [
        {
            "question_id": q1.question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
        },
        {
            "question_id": q2.question_id,
            "answer": (
                "The readiness gate forces active recall and implementation attempts first, "
                "which verifies conceptual understanding before code assistance."
            ),
        },
    ]

    payload = {
        "response_id": resp_id,
        "wants_full_code": True,
        "answers": answers,
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 200
    data = response.json()

    # FULL implementation permitted
    assert data["code_gate_decision"]["level"] == "FULL"
    assert data["full_code"] is not None
    assert "class TokenBucket" in data["full_code"]

    # Section for Complete Implementation added with explanation
    titles = [sec["title"] for sec in data["explanation"]["sections"]]
    assert "Complete Implementation" in titles


# ── Test 5: Failing Answers Return REQUIRE_CHECK ──────────────────────────────

@pytest.mark.asyncio
async def test_failing_answers_return_require_check():
    """Submitting answers with insufficient token overlap returns REQUIRE_CHECK."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    q1 = initial_response.questions[0]

    # Irrelevant answer
    answers = [
        {
            "question_id": q1.question_id,
            "answer": "I do not know anything about this topic.",
        }
    ]

    payload = {
        "response_id": resp_id,
        "wants_full_code": True,
        "answers": answers,
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data["full_code"] is None
    assert len(data["questions"]) >= 1


# ── Test 6: Unknown response_id Returns 404 ───────────────────────────────────

@pytest.mark.asyncio
async def test_unknown_response_id_returns_404():
    """POST /api/code-request with nonexistent response_id returns HTTP 404 Not Found."""
    payload = {
        "response_id": "non-existent-uuid-9999",
        "wants_full_code": True,
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 404
    assert "non-existent-uuid-9999" in response.json()["detail"]


# ── Test 7: Code Gate Violation Blocks Response ───────────────────────────────

@pytest.mark.asyncio
async def test_code_gate_violation_blocks_response():
    """Candidate code violating policy constraints is blocked by the code gate and response validator."""
    # 1. Policy check: candidate code with no explanation is flagged with violations
    violating_decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        readiness=LearnerReadiness(understanding_check_passed=True),
        proposed_full_code="class Leak: pass",
        has_explanation=False,  # Policy violation: code without explanation
    )
    assert violating_decision.level == CodeGateLevel.EXPLAIN
    assert len(violating_decision.violations) >= 1
    assert any("pedagogical explanation" in v for v in violating_decision.violations)

    # 2. Response validator check: code returning under EXPLAIN or REQUIRE_CHECK fails validation
    invalid_response = LearningResponse(
        learner_profile=LearnerProfile(),
        code_gate_decision=CodeGateDecision(level=CodeGateLevel.EXPLAIN),
        explanation=Explanation(
            topic="Test",
            summary="Summary",
            sections=[
                ExplanationSection(
                    title="Unauthorized Code",
                    content="Here is code under EXPLAIN: ```python\nx = 1\n```",
                    citation_resource_ids=[],
                )
            ],
            confidence_score=0.9,
        ),
    )
    report = validate_response(invalid_response)
    # Blocked by validator
    assert report.is_valid is False
    assert report.code_gate_respected is False
    assert any("No code may be returned under EXPLAIN" in issue for issue in report.issues)
