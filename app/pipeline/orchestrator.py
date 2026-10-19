"""
app/pipeline/orchestrator.py

Central pipeline orchestrator for DEVSCOUT Learning Companion.
Coordinates stages 1-9 end-to-end:
1. Developer Goal & Skill Context
2. Intent & Learning Plan
3. Research & Source Discovery
4. Evidence Review & Provenance Validation (with 4 -> 3 retry loop)
5. Explanation & Concept Mapping (grounded in verified source excerpts)
6. Guided Project Work
7. Understanding Check (conceptual questions with expected points)
8. Code Assistance Gate (EXPLAIN by default, SNIPPET if hint level allows)
9. Progress & Reflection (persisting learning history)
Cross-cutting: Response Validator & Repair, graceful degradation for optional stages.
"""
from __future__ import annotations

from datetime import UTC, datetime
import httpx
import structlog

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    CodeGateResult,
    CodeRequest,
    EvidenceReviewResult,
    ExperienceLevel,
    Explanation,
    ExplanationResult,
    ExplanationSection,
    FullCodeSolution,
    GoalContextResult,
    GuidedWorkResult,
    IntentPlanResult,
    LearnerProfile,
    LearningGoal,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    Milestone,
    ProgressRecord,
    ProgressReflectionResult,
    QuestionType,
    Resource,
    ResourceType,
    StageMetrics,
    UnderstandingCheckResult,
    UnderstandingQuestion,
    UnderstandingResult,
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
from app.pipeline.response_validator import repair_response, validate_response
from app.pipeline.session_store import (
    SessionData,
    SessionStore,
    create_session_from_response,
    get_session_store,
)
from app.pipeline.stage7_output_validation import evaluate_answers
from app.pipeline.structured_llm import generate_structured_with_retry
from app.policy.attempt_review import review_attempt
from app.policy.code_gate import LearnerReadiness
from app.research.github import GitHubConnector

logger = structlog.get_logger(__name__)


class SessionNotFoundError(Exception):
    """Raised when a requested response_id does not exist in the session store."""

    def __init__(self, response_id: str) -> None:
        super().__init__(f"Learning session not found for response_id: '{response_id}'")
        self.response_id = response_id


def _fallback_explanation_result(
    plan_result: IntentPlanResult,
    evidence_result: EvidenceReviewResult | None,
) -> ExplanationResult:
    """Create a graceful fallback explanation if Stage 5 fails."""
    topic = (
        plan_result.plan.milestones[0].title
        if plan_result.plan and plan_result.plan.milestones
        else "System Architecture"
    )
    first_res_id = (
        evidence_result.reviewed_resources[0].resource_id
        if evidence_result and evidence_result.reviewed_resources
        else ""
    )
    cites = [first_res_id] if first_res_id else []
    content = (
        f"Core conceptual foundations for {topic}. "
        "Review official documentation and architectural patterns to understand design trade-offs."
    )
    if not cites:
        content += " [Not backed by sources]"

    return ExplanationResult(
        explanation=Explanation(
            topic=topic,
            summary=f"Conceptual architecture and foundations for {topic}.",
            sections=[
                ExplanationSection(
                    title=f"Architecture Overview: {topic}",
                    content=content,
                    citation_resource_ids=cites,
                )
            ],
            confidence_score=0.85,
        ),
        concept_nodes=[],
        concept_edges=[],
    )


def _fallback_understanding_check_result(
    explanation_result: ExplanationResult,
) -> UnderstandingCheckResult:
    """Create fallback understanding questions if Stage 7 fails."""
    topic = explanation_result.explanation.topic if explanation_result.explanation else "System Architecture"
    return UnderstandingCheckResult(
        questions=[
            UnderstandingQuestion(
                concept_id="core",
                question_text=f"Why do we avoid untyped dictionaries across architectural boundaries in {topic}?",
                question_type=QuestionType.CONCEPTUAL,
                expected_points=[
                    "Type safety and static checking",
                    "Schema validation at boundaries",
                    "Self-documenting domain models",
                ],
            ),
            UnderstandingQuestion(
                concept_id="core",
                question_text="Explain in your own words how the readiness gate prevents passive copy-pasting.",
                question_type=QuestionType.EXPLAIN_BACK,
                expected_points=[
                    "Forces active recall or implementation attempt first",
                    "Verifies conceptual understanding before code assistance",
                ],
            ),
        ],
        results=[],
    )


async def run_learning_pipeline(
    request: LearnRequest,
    provider: GeminiProvider | None = None,
    github_connector: GitHubConnector | None = None,
    http_client: httpx.AsyncClient | None = None,
    session_store: SessionStore | None = None,
) -> LearningResponse:
    """
    Run pipeline end-to-end:
    Stage 1 -> 2 -> 3 -> 4 (with evidence retry loop) -> 5 -> 6 -> 7 -> 8 -> 9 -> validator.
    Missing or failed optional stages degrade gracefully and never 500.
    """
    cfg = get_settings()
    metrics: list[StageMetrics] = []

    # ── Stage 1: Developer Goal & Skill Context ───────────────────────────────
    goal_text = request.goal.strip()
    goal_title = goal_text.split("\n")[0][:100] if goal_text else "Developer Goal"
    goal_description = goal_text if len(goal_text) >= 10 else f"Learn and implement {goal_text}"

    stage1_result = await stage1_developer_input.run(
        goal_title=goal_title,
        goal_description=goal_description,
        experience_level=request.experience_level,
        tech_stack=request.tech_stack,
        constraints=request.constraints,
        current_project=request.project,
        target_skills=request.target_skills,
    )
    if stage1_result.metrics:
        metrics.append(stage1_result.metrics)

    # ── Stage 2: Intent & Learning Plan ───────────────────────────────────────
    depth = (
        "intermediate"
        if request.experience_level == ExperienceLevel.INTERMEDIATE
        else ("advanced" if request.experience_level == ExperienceLevel.ADVANCED else "beginner")
    )
    stage2_result = await stage2_input_processing.run(
        goal_context=stage1_result,
        depth_level=depth,
        provider=provider,
    )
    if stage2_result.metrics:
        metrics.append(stage2_result.metrics)

    # ── Stage 3 & 4: Research Discovery & Evidence Review Loop ────────────────
    max_loops = cfg.max_evidence_retry_loops
    loop_count = 0
    exclude_urls: list[str] = []
    evidence_result: EvidenceReviewResult | None = None

    while loop_count <= max_loops:
        stage3_result = await stage3_research.run(
            plan_result=stage2_result,
            loop_count=loop_count,
            exclude_urls=exclude_urls,
            provider=provider,
            github_connector=github_connector,
            http_client=http_client,
        )
        if stage3_result.metrics:
            metrics.append(stage3_result.metrics)

        evidence_result = await stage4_evidence_validation.run(
            discovery_result=stage3_result,
            loop_count=loop_count,
            max_retry_loops=max_loops,
        )
        if evidence_result.metrics:
            metrics.append(evidence_result.metrics)

        # Accumulate seen URLs to prevent duplicate scraping on retries
        for u in evidence_result.seen_urls:
            if u not in exclude_urls:
                exclude_urls.append(u)

        if evidence_result.is_sufficient:
            break

        if loop_count < max_loops:
            loop_count += 1
            logger.info(
                "evidence_insufficient_triggering_loop",
                loop_count=loop_count,
                seen_count=len(exclude_urls),
            )
        else:
            logger.warning(
                "evidence_insufficient_max_loops_reached_marking_uncertainty",
                loop_count=loop_count,
            )
            for r in evidence_result.reviewed_resources:
                r.uncertainty_flag = True
            break

    # ── Stage 5: Explanation & Concept Mapping ────────────────────────────────
    try:
        stage5_result = await stage5_ai_analysis.run(
            evidence_review=evidence_result,
            plan_result=stage2_result,
            provider=provider,
            auto_repair=True,
        )
    except Exception as exc:
        logger.warning("stage5_failed_degrading_gracefully", error=str(exc))
        stage5_result = _fallback_explanation_result(stage2_result, evidence_result)

    if stage5_result.metrics:
        metrics.append(stage5_result.metrics)

    # ── Stage 6: Guided Project Work (Optional) ───────────────────────────────
    stage6_result: GuidedWorkResult | None = None
    try:
        stage6_result = await stage6_recommendation.run(
            explanation_result=stage5_result,
            plan_result=stage2_result,
        )
        if stage6_result.metrics:
            metrics.append(stage6_result.metrics)
    except Exception as exc:
        logger.warning("stage6_failed_degrading_gracefully", error=str(exc))

    # ── Stage 7: Understanding Check (2-3 questions with expected_points) ─────
    try:
        stage7_result = await stage7_output_validation.run(
            explanation_result=stage5_result,
        )
        if stage7_result.metrics:
            metrics.append(stage7_result.metrics)
    except Exception as exc:
        logger.warning("stage7_failed_degrading_gracefully", error=str(exc))
        stage7_result = _fallback_understanding_check_result(stage5_result)

    # ── Stage 8: Code Assistance Gate ─────────────────────────────────────────
    # Default is EXPLAIN; SNIPPET only if hint level / request allows
    wants_snippet = request.wants_snippet
    try:
        stage8_result = await stage8_decision_report.run(
            explicit_request=False,
            wants_full_code=False,
            wants_snippet=wants_snippet,
            has_explanation=True,
        )
        if stage8_result.metrics:
            metrics.append(stage8_result.metrics)
    except Exception as exc:
        logger.warning("stage8_failed_degrading_gracefully", error=str(exc))
        stage8_result = CodeGateResult(
            decision=CodeGateDecision(
                level=CodeGateLevel.EXPLAIN,
                reasons=["Default pedagogical policy: explain concepts."],
            )
        )

    # ── Stage 9: Progress & Reflection (Optional) ─────────────────────────────
    progress_record: ProgressRecord | None = None
    try:
        completed_topics = [
            m.title for m in stage2_result.plan.milestones if m.is_completed
        ]
        stage9_result = await stage9_developer_feedback.run(
            learner_id=stage1_result.profile.learner_id,
            goal_id=stage1_result.goal.goal_id,
            completed_topics=completed_topics,
        )
        if stage9_result.metrics:
            metrics.append(stage9_result.metrics)
        progress_record = stage9_result.progress
    except Exception as exc:
        logger.warning("stage9_failed_degrading_gracefully", error=str(exc))

    # ── Assemble LearningResponse ─────────────────────────────────────────────
    response = LearningResponse(
        learner_profile=stage1_result.profile,
        learning_plan=stage2_result.plan,
        resources=evidence_result.reviewed_resources if evidence_result else [],
        explanation=stage5_result.explanation,
        snippets=stage8_result.allowed_snippets,
        concept_nodes=stage5_result.concept_nodes,
        concept_edges=stage5_result.concept_edges,
        questions=stage7_result.questions if stage7_result else [],
        understanding_result=(
            stage7_result.results[0]
            if (stage7_result and stage7_result.results)
            else None
        ),
        code_gate_decision=stage8_result.decision,
        progress=progress_record,
        confidence_score=(
            stage5_result.explanation.confidence_score
            if stage5_result.explanation
            else 0.9
        ),
        quality_score=0.92,
        metrics=metrics,
    )

    # ── Response Validator & Repair ───────────────────────────────────────────
    report = validate_response(response)
    if not report.is_valid:
        repaired_resp, _ = repair_response(response, report)
        response = repaired_resp

    # ── Persist to SessionStore ───────────────────────────────────────────────
    store = session_store or get_session_store()
    session = create_session_from_response(response)
    await store.save(session)

    return response


async def handle_code_request(
    request: CodeRequest,
    session_store: SessionStore | None = None,
    provider: GeminiProvider | None = None,
) -> LearningResponse:
    """
    Handle POST /api/code-request:
    - Look up session by response_id (or raise SessionNotFoundError).
    - Evaluate answers DETERMINISTICALLY using normalized token match against expected_points.
    - Review attempt_code against session's snippet hashes.
    - Build LearnerReadiness and evaluate Stage 8 Code Gate.
    - If level is FULL: make one structured Gemini call for full code + explanation,
      validate, and return it.
    - Otherwise: return REQUIRE_CHECK with reasons and questions.
    """
    store = session_store or get_session_store()
    session = await store.get(request.response_id)
    if not session:
        raise SessionNotFoundError(request.response_id)

    cfg = get_settings()

    # 1. Deterministic Answer Evaluation (Heuristic)
    understanding_check_passed = False
    if request.answers:
        eval_results = evaluate_answers(
            answers=request.answers,
            questions=session.learning_response.questions,
            threshold=cfg.answer_coverage_threshold,
        )
        session.understanding_results.extend(eval_results)
        if eval_results:
            session.learning_response.understanding_result = eval_results[-1]
            # Passed if all submitted answers met coverage threshold
            understanding_check_passed = all(r.passed for r in eval_results)

    # 2. Attempt Code Review
    attempt_review_res = None
    if request.attempt_code:
        attempt_review_res = review_attempt(
            code=request.attempt_code,
            language=request.attempt_language or "python",
            previous_snippet_hashes=session.snippet_hashes,
        )
        session.attempt_reviews.append(attempt_review_res)

    # 3. Build LearnerReadiness
    readiness = LearnerReadiness(
        understanding_check_passed=understanding_check_passed,
        attempt_review=attempt_review_res,
    )

    # 4. Call Stage 8 Code Gate
    gate_result = await stage8_decision_report.run(
        explicit_request=request.wants_full_code,
        wants_full_code=request.wants_full_code,
        readiness=readiness,
        has_explanation=True,
    )

    # 5. Handle Gate Outcomes
    if gate_result.decision.level == CodeGateLevel.FULL:
        if provider is None:
            provider = GeminiProvider()

        topic = (
            session.learning_response.explanation.topic
            if session.learning_response.explanation
            else "Implementation"
        )
        tech_stack = ", ".join(session.learning_response.learner_profile.tech_stack) or "Python"

        prompt = (
            f"You are a principal software engineer. The learner has demonstrated readiness.\n"
            f"Provide the complete, verified, runnable implementation for:\n"
            f"Topic: {topic}\n"
            f"Tech Stack: {tech_stack}\n\n"
            f"Requirements:\n"
            f"1. Complete, production-grade, runnable code.\n"
            f"2. Comprehensive pedagogical explanation of the architecture.\n"
            f"3. Line-by-line explanation of key implementation details.\n"
        )
        system_prompt = (
            "You are an expert technical mentor. Generate production-grade code with in-depth "
            "architectural and line-by-line explanations. Adhere strictly to the requested schema."
        )

        solution = await generate_structured_with_retry(
            provider=provider,
            prompt=prompt,
            schema=FullCodeSolution,
            system_prompt=system_prompt,
        )

        session.learning_response.full_code = solution.code
        session.learning_response.code_gate_decision = gate_result.decision

        if session.learning_response.explanation:
            first_res_id = (
                session.learning_response.resources[0].resource_id
                if session.learning_response.resources
                else ""
            )
            cites = [first_res_id] if first_res_id else []
            lines_text = "\n".join(f"- {line}" for line in solution.line_by_line_explanation)
            content = (
                f"{solution.explanation}\n\n"
                f"```python\n{solution.code}\n```\n\n"
                f"### Line-by-Line Breakdown\n{lines_text}"
            )
            session.learning_response.explanation.sections.append(
                ExplanationSection(
                    title="Complete Implementation",
                    content=content,
                    citation_resource_ids=cites,
                )
            )

        # Validate through response_validator
        report = validate_response(session.learning_response)
        if not report.is_valid:
            repaired_resp, _ = repair_response(session.learning_response, report)
            session.learning_response = repaired_resp

    else:
        # Denied full code: return REQUIRE_CHECK or EXPLAIN with reasons and questions
        session.learning_response.full_code = None
        session.learning_response.code_gate_decision = gate_result.decision

    await store.save(session)
    return session.learning_response
