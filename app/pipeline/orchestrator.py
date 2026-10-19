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

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
import re
import httpx
import structlog

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    AttemptStatus,
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
    SourceDiscoveryResult,
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
from app.pipeline.stage7_output_validation import (
    check_understanding_completion,
    evaluate_answers,
)
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
    on_progress: Callable[[int, str, float], Awaitable[None] | None] | None = None,
) -> LearningResponse:
    """
    Run pipeline end-to-end:
    Stage 1 -> 2 -> 3 -> 4 (with evidence retry loop) -> 5 -> 6 -> 7 -> 8 -> 9 -> validator.
    Missing or failed optional stages degrade gracefully and never 500.
    """
    cfg = get_settings()
    metrics: list[StageMetrics] = []

    async def _report_stage(stage_id: int, stage_name: str, pct: float) -> None:
        if on_progress:
            try:
                res = on_progress(stage_id, stage_name, pct)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    # ── Stage 1: Developer Goal & Skill Context ───────────────────────────────
    await _report_stage(1, "Stage 1: Developer Goal & Skill Context", 1 / 9)
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
    await _report_stage(2, "Stage 2: Intent & Learning Plan", 2 / 9)
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
    accumulated_resources: list[Resource] = []
    plan_topics = stage3_research.extract_topics(stage2_result)
    only_topics: list[str] | None = None

    while loop_count <= max_loops:
        await _report_stage(3, "Stage 3: Research Discovery", 3 / 9)
        stage3_result = await stage3_research.run(
            plan_result=stage2_result,
            loop_count=loop_count,
            exclude_urls=exclude_urls,
            only_topics=only_topics,
            provider=provider,
            github_connector=github_connector,
            http_client=http_client,
        )
        if stage3_result.metrics:
            metrics.append(stage3_result.metrics)

        # Merge discovered resources with previously accumulated resources (deduped by URL)
        seen_res_urls = {r.url for r in accumulated_resources if r.url}
        for r in stage3_result.resources:
            if r.url and r.url not in seen_res_urls:
                accumulated_resources.append(r)
                seen_res_urls.add(r.url)
            elif not r.url:
                accumulated_resources.append(r)

        combined_discovery = SourceDiscoveryResult(
            goal_id=stage3_result.goal_id,
            resources=list(accumulated_resources),
            loop_count=loop_count,
            metrics=stage3_result.metrics,
        )

        await _report_stage(4, "Stage 4: Evidence Review & Validation", 4 / 9)
        evidence_result = await stage4_evidence_validation.run(
            discovery_result=combined_discovery,
            loop_count=loop_count,
            max_retry_loops=max_loops,
            expected_topics=plan_topics,
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
            only_topics = (
                evidence_result.insufficient_topics
                if evidence_result.insufficient_topics
                else None
            )
            logger.info(
                "evidence_insufficient_triggering_loop",
                loop_count=loop_count,
                seen_count=len(exclude_urls),
                only_topics=only_topics,
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
    await _report_stage(5, "Stage 5: Explanation & Concept Mapping", 5 / 9)
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
    await _report_stage(6, "Stage 6: Guided Project Work", 6 / 9)
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
    await _report_stage(7, "Stage 7: Understanding Check", 7 / 9)
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
    await _report_stage(8, "Stage 8: Code Assistance Gate", 8 / 9)
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
    await _report_stage(9, "Stage 9: Progress & Reflection", 9 / 9)
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

    raw_questions = stage7_result.questions if stage7_result else []

    # ── Assemble LearningResponse ─────────────────────────────────────────────
    response = LearningResponse(
        learner_profile=stage1_result.profile,
        learning_plan=stage2_result.plan,
        resources=evidence_result.reviewed_resources if evidence_result else [],
        explanation=stage5_result.explanation,
        snippets=stage8_result.allowed_snippets,
        concept_nodes=stage5_result.concept_nodes,
        concept_edges=stage5_result.concept_edges,
        questions=raw_questions,
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
    session = create_session_from_response(
        response,
        questions=raw_questions,
        goal_title=goal_title,
        goal_description=goal_description,
    )
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
    - Req 1: Evaluate answers requiring ALL questions answered (or min_questions_answered),
      every answer must meet the coverage threshold.
    - Req 2: Persist understanding_passed in session; build LearnerReadiness from
      session state (passed check OR any ACCEPTABLE attempt review in history).
    - If level is FULL: generate code grounded in learner goal/project/milestones/stack,
      validate Python with ast.parse (retry once on failure), set code_notice.
    - Otherwise: return REQUIRE_CHECK with reasons and questions.
    """
    store = session_store or get_session_store()
    session = await store.get(request.response_id)
    if not session:
        raise SessionNotFoundError(request.response_id)

    cfg = get_settings()
    understanding_reason: str | None = None

    # ── Req 1 & 2: Deterministic Answer Evaluation & Attempt Tracking ─────────
    if request.answers:
        now = datetime.now(UTC)

        # Check attempt cooldown
        if session.last_understanding_attempt_at is not None and cfg.attempt_cooldown_seconds > 0:
            elapsed = (now - session.last_understanding_attempt_at).total_seconds()
            if elapsed < cfg.attempt_cooldown_seconds:
                remaining = int(cfg.attempt_cooldown_seconds - elapsed) + 1
                understanding_reason = (
                    f"Understanding check cooldown active. Please wait {remaining} seconds before retrying."
                )
                session.understanding_passed = False
                logger.info("understanding_check_cooldown_active", remaining=remaining)

        # Check maximum attempts cap
        if understanding_reason is None and session.understanding_attempt_count >= cfg.max_understanding_attempts:
            understanding_reason = (
                f"Maximum understanding check attempts ({cfg.max_understanding_attempts}) exceeded for this session."
            )
            session.understanding_passed = False
            logger.info("understanding_check_max_attempts_exceeded", count=session.understanding_attempt_count)

        if understanding_reason is None:
            session.understanding_attempt_count += 1
            session.last_understanding_attempt_at = now

            questions_to_eval = session.questions if session.questions else session.learning_response.questions
            eval_results = evaluate_answers(
                answers=request.answers,
                questions=questions_to_eval,
                threshold=cfg.answer_coverage_threshold,
            )
            session.understanding_results.extend(eval_results)
            if eval_results:
                session.learning_response.understanding_result = eval_results[-1]

            # Map each answered question to its latest evaluation result in this session
            latest_by_qid: dict[str, UnderstandingResult] = {}
            for r in session.understanding_results:
                latest_by_qid[r.question_id] = r

            answered_count = len(latest_by_qid)
            total_questions = len(questions_to_eval)
            all_passed = (
                len(latest_by_qid) > 0
                and all(r.passed for r in latest_by_qid.values())
            )
            completion_ok, completion_err = check_understanding_completion(
                answered_count=answered_count,
                total_questions=total_questions,
                min_questions_answered=cfg.min_questions_answered,
                min_ratio=0.60,
            )
            if completion_ok and all_passed:
                session.understanding_passed = True
                logger.info(
                    "understanding_check_passed",
                    answered=answered_count,
                    required=cfg.min_questions_answered,
                    total=total_questions,
                )
            else:
                session.understanding_passed = False
                understanding_reason = completion_err or "One or more answers did not meet the coverage threshold."
                logger.info(
                    "understanding_check_failed",
                    answered=answered_count,
                    required=cfg.min_questions_answered,
                    total=total_questions,
                    all_passed=all_passed,
                    reason=understanding_reason,
                )
        else:
            session.learning_response.understanding_result = UnderstandingResult(
                question_id=request.answers[0].question_id if request.answers else "general",
                passed=False,
                learner_response=request.answers[0].answer if request.answers else "",
                feedback=understanding_reason,
                detected_misconceptions=[understanding_reason],
                score=0.0,
            )

    # ── Req 2: Attempt Code Review with Relevance & Snippet Similarity ─────────
    attempt_review_res = None
    if request.attempt_code:
        milestone_concepts: list[str] = []
        lr = session.learning_response
        if lr.learning_plan and lr.learning_plan.milestones:
            current_milestone = next(
                (m for m in lr.learning_plan.milestones if not m.is_completed),
                lr.learning_plan.milestones[0],
            )
            if current_milestone.concepts:
                milestone_concepts.extend(current_milestone.concepts)
            elif lr.concept_nodes:
                milestone_concepts.extend(node.label for node in lr.concept_nodes)
            else:
                words = [w for w in re.findall(r"\w+", current_milestone.title.lower()) if len(w) >= 3]
                milestone_concepts.extend(words)

        attempt_review_res = review_attempt(
            code=request.attempt_code,
            language=request.attempt_language or "python",
            previous_snippet_hashes=session.snippet_hashes,
            previous_snippets=session.snippets,
            milestone_concepts=milestone_concepts if milestone_concepts else None,
        )
        session.attempt_reviews.append(attempt_review_res)

    # ── Req 2: Build LearnerReadiness from session state (not just current request) ──
    # understanding_passed comes from the persisted session flag (survives between calls)
    # attempt_review comes from any ACCEPTABLE attempt ever submitted in this session
    session_understanding_passed = session.understanding_passed
    session_acceptable_attempt = next(
        (r for r in session.attempt_reviews if r.status == AttemptStatus.ACCEPTABLE),
        None,
    )

    active_unlocked_by: str | None = None
    if request.attempt_code and attempt_review_res and attempt_review_res.status == AttemptStatus.ACCEPTABLE:
        active_unlocked_by = "attempt"
    elif request.answers and session_understanding_passed:
        active_unlocked_by = "understanding"
    elif session_understanding_passed:
        active_unlocked_by = "understanding"
    elif session_acceptable_attempt:
        active_unlocked_by = "attempt"

    readiness = LearnerReadiness(
        understanding_check_passed=session_understanding_passed,
        attempt_review=session_acceptable_attempt or attempt_review_res,
        unlocked_by=active_unlocked_by,
    )

    # ── Stage 8 Code Gate ─────────────────────────────────────────────────────
    gate_result = await stage8_decision_report.run(
        explicit_request=request.wants_full_code,
        wants_full_code=request.wants_full_code,
        readiness=readiness,
        has_explanation=True,
    )

    if understanding_reason and gate_result.decision.level != CodeGateLevel.FULL:
        gate_result.decision.reasons.insert(0, understanding_reason)

    # ── Req 4: Full Code Generation ────────────────────────────────────────────
    if gate_result.decision.level == CodeGateLevel.FULL:
        if provider is None:
            provider = GeminiProvider()

        lr = session.learning_response
        topic = lr.explanation.topic if lr.explanation else "Implementation"
        goal_title = (
            session.goal_title
            or (lr.learning_plan.milestones[0].title if (lr.learning_plan and lr.learning_plan.milestones) else "")
            or topic
        )
        goal_desc = (
            session.goal_description
            or (lr.learning_plan.milestones[0].description if (lr.learning_plan and lr.learning_plan.milestones) else "")
        )
        milestones_text = ""
        if lr.learning_plan and lr.learning_plan.milestones:
            ms_items = "\n".join(
                f"  - {m.title}: {m.target_outcome}" for m in lr.learning_plan.milestones
            )
            milestones_text = f"\nMilestones:\n{ms_items}"
        tech_stack = ", ".join(lr.learner_profile.tech_stack) or "Python"
        current_project = lr.learner_profile.current_project or "personal project"

        # Req 4: no "verified" or "production-grade" in prompt
        prompt = (
            f"You are a principal software engineer mentoring a developer.\n"
            f"Goal: {goal_title}\n"
            f"Description: {goal_desc}\n"
            f"Current project: {current_project}\n"
            f"Tech stack: {tech_stack}{milestones_text}\n\n"
            f"Provide a complete, runnable implementation for the goal above.\n"
            f"Requirements:\n"
            f"1. Complete, well-structured, runnable code with clear docstrings.\n"
            f"2. Comprehensive explanation of the architecture and design decisions.\n"
            f"3. Line-by-line explanation of key implementation details.\n"
        )
        system_prompt = (
            "You are an expert technical mentor. Generate complete, well-structured code with "
            "in-depth architectural and line-by-line explanations. Adhere strictly to the requested schema."
        )

        solution = await generate_structured_with_retry(
            provider=provider,
            prompt=prompt,
            schema=FullCodeSolution,
            system_prompt=system_prompt,
        )

        # Req 4: For Python output, validate with ast.parse; retry once on failure
        code_to_use = solution.code
        if solution.language.lower() == "python":
            import ast as _ast

            try:
                _ast.parse(code_to_use)
            except SyntaxError as syn_err:
                logger.warning("ast_parse_failed_retrying", error=str(syn_err))
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"The previous attempt had a Python syntax error: {syn_err}\n"
                    f"Fix it and return valid, parseable Python code."
                )
                try:
                    retry_solution = await generate_structured_with_retry(
                        provider=provider,
                        prompt=retry_prompt,
                        schema=FullCodeSolution,
                        system_prompt=system_prompt,
                    )
                    _ast.parse(retry_solution.code)
                    solution = retry_solution
                    code_to_use = retry_solution.code
                    logger.info("ast_parse_retry_succeeded")
                except (SyntaxError, Exception) as retry_err:
                    logger.error("ast_parse_retry_failed", error=str(retry_err))
                    raise ValueError(
                        f"Generated Python code has a syntax error that could not be fixed: {retry_err}"
                    ) from retry_err

        session.learning_response.full_code = code_to_use
        # Req 4: set code_notice
        session.learning_response.code_notice = "Generated code, not executed or tested."
        session.learning_response.code_gate_decision = gate_result.decision
        session.learning_response.unlocked_by = gate_result.decision.unlocked_by

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
                f"```python\n{code_to_use}\n```\n\n"
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
        session.learning_response.code_notice = None
        session.learning_response.code_gate_decision = gate_result.decision
        session.learning_response.unlocked_by = None

    await store.save(session)
    return session.learning_response

