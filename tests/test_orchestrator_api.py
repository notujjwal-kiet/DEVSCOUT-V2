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
                explanation="Complete token bucket implementation.",
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

    # 2-3 understanding questions returned but expected_points MUST NOT exist in public questions
    assert len(data["questions"]) in (2, 3)
    assert "expected_points" not in response.text
    for q in data["questions"]:
        # Req 3: answer key never exposed to API consumers
        assert "expected_points" not in q, f"expected_points leaked for question {q['question_id']}"

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


@pytest.mark.asyncio
async def test_orchestrator_evidence_loop_reruns_only_thin_topics(monkeypatch):
    """When Stage 4 identifies thin/insufficient topics, Stage 3 is retried with only_topics for those topics."""
    from app.pipeline import stage3_research, stage4_evidence_validation

    call_counts = {"stage3": 0, "stage4": 0}
    stage3_received_only_topics: list[list[str] | None] = []

    real_stage3_run = stage3_research.run
    real_stage4_run = stage4_evidence_validation.run

    async def mock_stage3_tracking(*args, **kwargs):
        call_counts["stage3"] += 1
        only_t = kwargs.get("only_topics")
        stage3_received_only_topics.append(list(only_t) if only_t is not None else None)
        return await real_stage3_run(*args, **kwargs)

    async def mock_stage4_with_thin_topic(*args, **kwargs):
        call_counts["stage4"] += 1
        res = await real_stage4_run(*args, **kwargs)
        if call_counts["stage4"] == 1:
            # First attempt: only "Sliding Window" is thin/insufficient
            res.is_sufficient = False
            res.needs_more_evidence = True
            res.insufficient_topics = ["Sliding Window"]
        else:
            # Retry attempt: sufficient
            res.is_sufficient = True
            res.needs_more_evidence = False
            res.insufficient_topics = []
        return res

    monkeypatch.setattr("app.pipeline.stage3_research.run", mock_stage3_tracking)
    monkeypatch.setattr("app.pipeline.stage4_evidence_validation.run", mock_stage4_with_thin_topic)

    request = LearnRequest(
        goal="Distributed Rate Limiter",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )

    result = await run_learning_pipeline(request)

    assert call_counts["stage3"] == 2
    assert call_counts["stage4"] == 2

    # Loop 0 searched all topics (only_topics is None)
    assert stage3_received_only_topics[0] is None
    # Loop 1 retried ONLY the thin topic
    assert stage3_received_only_topics[1] == ["Sliding Window"]
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
    assert data["code_notice"] == "Generated code, not executed or tested."
    assert "expected_points" not in response.text

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


# ── Test 8: 1-of-3 Answers Does Not Pass ──────────────────────────────────────

@pytest.mark.asyncio
async def test_one_of_three_answers_does_not_pass():
    """Answering only 1 question when session has 2-3 questions must NOT pass the understanding check."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions
    assert len(questions) >= 2

    # Provide high-quality answer for ONLY the first question
    q1 = questions[0]
    answers = [
        {
            "question_id": q1.question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
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

    # 1 of 3 (or 1 of 2) answers must NOT pass
    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data["full_code"] is None

    # Session store must reflect understanding check did NOT pass
    store = get_session_store()
    session = await store.get(resp_id)
    assert session is not None
    assert session.understanding_passed is False


# ── Test 9: Two-Step Flow ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_two_step_flow_persists_readiness_and_unlocks_full():
    """Submit passing answers with wants_full_code=false, then request full code with no answers -> FULL."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions
    assert len(questions) >= 2

    passing_answers = [
        {
            "question_id": questions[0].question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
        },
        {
            "question_id": questions[1].question_id,
            "answer": (
                "The readiness gate forces active recall and implementation attempts first, "
                "which verifies conceptual understanding before code assistance."
            ),
        },
    ]

    # Step 1: Submit passing answers with wants_full_code=false
    step1_payload = {
        "response_id": resp_id,
        "wants_full_code": False,
        "answers": passing_answers,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        step1_resp = await client.post("/api/code-request", json=step1_payload)

    assert step1_resp.status_code == 200
    step1_data = step1_resp.json()
    assert step1_data["code_gate_decision"]["level"] == "EXPLAIN"
    assert step1_data["full_code"] is None

    # Verify understanding_passed is persisted in SessionData
    store = get_session_store()
    session_after_step1 = await store.get(resp_id)
    assert session_after_step1 is not None
    assert session_after_step1.understanding_passed is True

    # Step 2: Request full code with wants_full_code=true and NO answers
    step2_payload = {
        "response_id": resp_id,
        "wants_full_code": True,
        "answers": [],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        step2_resp = await client.post("/api/code-request", json=step2_payload)

    assert step2_resp.status_code == 200
    step2_data = step2_resp.json()
    assert step2_data["code_gate_decision"]["level"] == "FULL"
    assert step2_data["full_code"] is not None
    assert "class TokenBucket" in step2_data["full_code"]
    assert step2_data["code_notice"] == "Generated code, not executed or tested."


# ── Test 10: Wrong Answer Fails Understanding Check ────────────────────────────

@pytest.mark.asyncio
async def test_wrong_answer_fails_understanding_check():
    """Submitting answers where any answer fails coverage must NOT pass."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions

    # One good answer, one completely wrong answer
    answers = [
        {
            "question_id": questions[0].question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
        },
        {
            "question_id": questions[1].question_id,
            "answer": "Banana potato orange nonsense completely irrelevant.",
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

    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data["full_code"] is None

    store = get_session_store()
    session = await store.get(resp_id)
    assert session is not None
    assert session.understanding_passed is False


# ── Test 11: No Answers on Fresh Session ──────────────────────────────────────

@pytest.mark.asyncio
async def test_no_answers_returns_require_check():
    """Requesting full code with no answers on fresh session returns REQUIRE_CHECK."""
    init_request = LearnRequest(
        goal="Rate Limiter Implementation",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["Python"],
    )
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id

    payload = {
        "response_id": resp_id,
        "wants_full_code": True,
        "answers": [],
    }

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert data["code_gate_decision"]["level"] == "REQUIRE_CHECK"
    assert data["full_code"] is None

    store = get_session_store()
    session = await store.get(resp_id)
    assert session is not None
    assert session.understanding_passed is False


# ── Test 12: Expected Points Never Leaked to Client ───────────────────────────

@pytest.mark.asyncio
async def test_expected_points_leak_check():
    """The raw JSON of /api/learn and /api/code-request contains no 'expected_points'."""
    # 1. Check /api/learn response
    payload = {
        "goal": "Build a Distributed Rate Limiter",
        "experience_level": "intermediate",
        "tech_stack": ["Python"],
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        learn_resp = await client.post("/api/learn", json=payload)

    assert learn_resp.status_code == 200
    assert "expected_points" not in learn_resp.text
    learn_data = learn_resp.json()
    assert len(learn_data["questions"]) >= 1
    for q in learn_data["questions"]:
        assert "expected_points" not in q

    resp_id = learn_data["response_id"]

    # Verify expected_points IS kept internally in SessionData
    store = get_session_store()
    session = await store.get(resp_id)
    assert session is not None
    assert len(session.questions) >= 1
    for q in session.questions:
        assert len(q.expected_points) >= 1, "Answer key must be preserved in SessionData"

    # 2. Check /api/code-request response
    code_payload = {
        "response_id": resp_id,
        "wants_full_code": False,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        code_resp = await client.post("/api/code-request", json=code_payload)

    assert code_resp.status_code == 200
    assert "expected_points" not in code_resp.text
    code_data = code_resp.json()
    for q in code_data["questions"]:
        assert "expected_points" not in q


# ── Test 13: Python AST Parse Retry and Error ─────────────────────────────────

@pytest.mark.asyncio
async def test_python_code_ast_parse_retry_succeeds(monkeypatch):
    """If generated Python code has syntax error, retry once; if retry succeeds, return valid code."""
    attempts = 0

    async def fake_retry_generate(self, prompt, schema, system_prompt=None):
        nonlocal attempts
        if schema == FullCodeSolution:
            attempts += 1
            if attempts == 1:
                # Syntax error on first try
                return FullCodeSolution(
                    code="def invalid syntax here ( broken",
                    language="python",
                    explanation="Attempt 1 with syntax error.",
                    line_by_line_explanation=["broken line"],
                )
            else:
                # Valid code on retry
                return FullCodeSolution(
                    code="class TokenBucketFixed:\n    pass\n",
                    language="python",
                    explanation="Attempt 2 valid code.",
                    line_by_line_explanation=["Fixed class"],
                )
        if schema == LearningPlan:
            return LearningPlan(
                goal_id="g1",
                milestones=[Milestone(title="Milestone 1", description="Description 1", target_outcome="Outcome 1")],
            )
        from app.models.pipeline import ExplanationResult
        if schema == ExplanationResult:
            return ExplanationResult(
                explanation=Explanation(
                    topic="Topic",
                    summary="Summary",
                    sections=[ExplanationSection(title="T", content="Content", citation_resource_ids=["res-docs-1"])],
                )
            )
        raise ValueError(f"Unexpected schema: {schema}")

    monkeypatch.setattr("app.llm.provider.GeminiProvider.generate_structured", fake_retry_generate)

    init_request = LearnRequest(goal="Rate Limiter", tech_stack=["Python"])
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions

    passing_answers = [
        {
            "question_id": questions[0].question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
        },
        {
            "question_id": questions[1].question_id,
            "answer": (
                "The readiness gate forces active recall and implementation attempts first, "
                "which verifies conceptual understanding before code assistance."
            ),
        },
    ]

    payload = {"response_id": resp_id, "wants_full_code": True, "answers": passing_answers}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["code_gate_decision"]["level"] == "FULL"
    assert "class TokenBucketFixed" in data["full_code"]
    assert data["code_notice"] == "Generated code, not executed or tested."
    assert attempts == 2


@pytest.mark.asyncio
async def test_python_code_ast_parse_retry_fails_cleanly(monkeypatch):
    """If generated Python code has syntax error and retry also fails, handle_code_request fails clearly."""
    async def fake_always_broken(self, prompt, schema, system_prompt=None):
        if schema == FullCodeSolution:
            return FullCodeSolution(
                code="def permanently broken (",
                language="python",
                explanation="Broken code.",
                line_by_line_explanation=["broken line"],
            )
        if schema == LearningPlan:
            return LearningPlan(
                goal_id="g1",
                milestones=[Milestone(title="Milestone 1", description="Description 1", target_outcome="Outcome 1")],
            )
        from app.models.pipeline import ExplanationResult
        if schema == ExplanationResult:
            return ExplanationResult(
                explanation=Explanation(
                    topic="Topic",
                    summary="Summary",
                    sections=[ExplanationSection(title="T", content="Content", citation_resource_ids=["res-docs-1"])],
                )
            )
        raise ValueError(f"Unexpected schema: {schema}")

    monkeypatch.setattr("app.llm.provider.GeminiProvider.generate_structured", fake_always_broken)

    init_request = LearnRequest(goal="Rate Limiter", tech_stack=["Python"])
    initial_response = await run_learning_pipeline(init_request)
    resp_id = initial_response.response_id
    questions = initial_response.questions

    passing_answers = [
        {
            "question_id": questions[0].question_id,
            "answer": (
                "We use type safety and static checking to enforce schema validation "
                "at architectural boundaries, creating self-documenting domain models."
            ),
        },
        {
            "question_id": questions[1].question_id,
            "answer": (
                "The readiness gate forces active recall and implementation attempts first, "
                "which verifies conceptual understanding before code assistance."
            ),
        },
    ]

    # When called via API, returns generic 500 error
    payload = {"response_id": resp_id, "wants_full_code": True, "answers": passing_answers}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/code-request", json=payload)

    assert response.status_code == 500
    assert response.json()["detail"] == "An internal error occurred. Please try again later."


# ── Test 14: Generic 500 Message on Internal Server Error ─────────────────────

@pytest.mark.asyncio
async def test_generic_500_message_on_internal_error(monkeypatch):
    """Both /api/learn and /api/code-request return generic message on 500 without leaking exception internals."""
    async def crash_pipeline(*args, **kwargs):
        raise RuntimeError("Confidential database connection failed: secret_db_pw=12345")

    monkeypatch.setattr("app.api.learn.run_learning_pipeline", crash_pipeline)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp1 = await client.post("/api/learn", json={"goal": "Crashing test"})

    assert resp1.status_code == 500
    data1 = resp1.json()
    assert data1["detail"] == "An internal error occurred. Please try again later."
    assert "secret_db_pw" not in resp1.text

    async def crash_code_request(*args, **kwargs):
        raise RuntimeError("Confidential token leakage: internal_token=xyz999")

    monkeypatch.setattr("app.api.learn.handle_code_request", crash_code_request)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp2 = await client.post("/api/code-request", json={"response_id": "any-id", "wants_full_code": True})

    assert resp2.status_code == 500
    data2 = resp2.json()
    assert data2["detail"] == "An internal error occurred. Please try again later."
    assert "internal_token" not in resp2.text

