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
import time
import uuid
import httpx
import structlog

from app.config import SEED_DOCS, get_settings
from app.models.pipeline import (
    AttemptStatus,
    CodeGateDecision,
    CodeGateLevel,
    CodeGateResult,
    CodeRequest,
    EvidenceReviewResult,
    EvidenceSummary,
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
    RejectedSource,
    Resource,
    ResourceType,
    ResponseWarning,
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
from app.pipeline.response_validator import (
    calibrate_confidence_score,
    repair_response,
    strip_unbacked_marker,
    validate_response,
)
from app.pipeline.session_store import (
    SessionData,
    SessionStore,
    create_session_from_response,
    get_session_store,
)
from app.pipeline.stage3_research import extract_clean_excerpt, fetch_page
from app.pipeline.stage7_output_validation import (
    check_understanding_completion,
    evaluate_answers,
)
from app.pipeline.structured_llm import generate_structured_with_retry
from app.policy.attempt_review import review_attempt
from app.policy.code_gate import LearnerReadiness
from app.research.github import GitHubConnector
from app.research.models import ResearchDocument
from app.research.scoring import (
    hash_content,
    is_source_relevant_to_goal,
    score_document,
)
import threading

logger = structlog.get_logger(__name__)

_session_locks: dict[tuple[int, str], asyncio.Lock] = {}
_session_locks_guard = threading.Lock()


def _get_session_lock(response_id: str) -> asyncio.Lock:
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
    except RuntimeError:
        loop_id = 0
    key = (loop_id, response_id)
    with _session_locks_guard:
        if key not in _session_locks:
            _session_locks[key] = asyncio.Lock()
        return _session_locks[key]


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

    return ExplanationResult(
        explanation=Explanation(
            topic=topic,
            summary=f"Conceptual architecture and foundations for {topic}.",
            sections=[
                ExplanationSection(
                    title=f"Architecture Overview: {topic}",
                    content=content,
                    citation_resource_ids=cites,
                    support_status="supported" if cites else "unsupported",
                )
            ],
            confidence_score=0.85,
        ),
        concept_nodes=[],
        concept_edges=[],
    )


def compute_quality_score(
    response: LearningResponse,
    evidence_status: str,
) -> float:
    """
    Compute quality_score deterministically without model guessing:
    Formula:
      quality_score = 0.5 * structure_complete + 0.5 * fraction_supported

    Components:
      - structure_complete: 1.0 if schema is valid, explanation sections are present,
        and questions are present; 0.0 otherwise.
      - fraction_supported: fraction of explanation sections where support_status == "supported"
        (i.e. supported_sections / total_sections if total_sections > 0 else 0.0).

    Cap:
      - If evidence_status is "insufficient", quality_score is capped at 0.4:
        quality_score = min(0.4, quality_score).
    """
    sections = response.explanation.sections if response.explanation else []
    total_sections = len(sections)
    supported_sections = sum(1 for s in sections if s.support_status == "supported")
    fraction_supported = (supported_sections / total_sections) if total_sections > 0 else 0.0

    sections_present = total_sections > 0
    questions_present = len(response.questions) > 0
    schema_valid = bool(
        response.learner_profile
        and response.learning_plan
        and response.explanation
        and response.code_gate_decision
    )

    structure_complete = 1.0 if (schema_valid and sections_present and questions_present) else 0.0

    raw_score = 0.5 * structure_complete + 0.5 * fraction_supported

    if evidence_status == "insufficient":
        raw_score = min(0.4, raw_score)

    return round(raw_score, 3)


def _fallback_understanding_check_result(
    explanation_result: ExplanationResult,
) -> UnderstandingCheckResult:
    """Create fallback understanding questions if Stage 7 fails."""
    topic = (
        explanation_result.explanation.topic
        if explanation_result.explanation
        else "System Architecture"
    )
    concept_nodes = explanation_result.concept_nodes or []
    fallback_qs = stage7_output_validation.build_fallback_questions(topic, concept_nodes)
    return UnderstandingCheckResult(
        questions=fallback_qs,
        results=[],
    )


async def run_learning_pipeline(
    request: LearnRequest,
    provider: object | None = None,
    github_connector: GitHubConnector | None = None,
    http_client: httpx.AsyncClient | None = None,
    session_store: SessionStore | None = None,
    on_progress: Callable[[int, str, float], Awaitable[None] | None] | None = None,
    use_cache: bool = False,
) -> LearningResponse:
    """
    Run pipeline end-to-end:
    Stage 1 -> 2 -> 3 -> 4 (with evidence retry loop) -> 5 -> 6 -> 7 -> 8 -> 9 -> validator.
    Missing or failed optional stages degrade gracefully and never 500.
    """
    cfg = get_settings()
    if use_cache and not request.refresh:
        from app.pipeline.request_cache import get_cached_learning_response

        cached = get_cached_learning_response(request)
        if cached is not None:
            logger.info("orchestrator_request_cache_hit", goal=request.goal)
            store = session_store or get_session_store()
            goal_text = request.goal.strip()
            goal_title = goal_text.split("\n")[0][:100] if goal_text else "Developer Goal"
            goal_description = (
                goal_text if len(goal_text) >= 10 else f"Learn and implement {goal_text}"
            )
            session = create_session_from_response(
                cached,
                questions=cached.questions,
                goal_title=goal_title,
                goal_description=goal_description,
            )
            await store.save(session)
            return cached

    metrics: list[StageMetrics] = []

    def _record_metrics(m: StageMetrics | None) -> None:
        if m is not None:
            metrics.append(m)
            logger.info(
                "stage_duration",
                stage_id=m.stage_id,
                stage_name=m.stage_name,
                duration_ms=round(m.latency_ms, 2),
                prompt_tokens=m.prompt_tokens,
                completion_tokens=m.completion_tokens,
                total_tokens=m.total_tokens,
            )

    async def _report_stage(stage_id: int, stage_name: str, pct: float) -> None:
        if on_progress:
            try:
                res = on_progress(stage_id, stage_name, pct)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                pass

    # ── Stage 1: Developer Goal & Skill Context ───────────────────────────────
    logger.info("stage 1 start")
    _t1 = time.perf_counter()
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
    _record_metrics(stage1_result.metrics)
    logger.info(f"stage 1 done in {time.perf_counter() - _t1:.1f}s")

    # ── Stage 2: Intent & Learning Plan ───────────────────────────────────────
    logger.info("stage 2 start")
    _t2 = time.perf_counter()
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
        requested_model_id=getattr(request, "model_id", None),
    )
    # Ensure server-generated plan_id and created_at exist
    if stage2_result.plan:
        if not getattr(stage2_result.plan, "plan_id", None):
            stage2_result.plan.plan_id = str(uuid.uuid4())
        if not getattr(stage2_result.plan, "created_at", None):
            stage2_result.plan.created_at = datetime.now(UTC)

    _record_metrics(stage2_result.metrics)
    logger.info(f"stage 2 done in {time.perf_counter() - _t2:.1f}s")

    # ── Stage 3 & 4: Research Discovery & Evidence Review Loop ────────────────
    max_loops = cfg.max_evidence_retry_loops
    loop_count = 0
    exclude_urls: list[str] = []
    evidence_result: EvidenceReviewResult | None = None
    accumulated_resources: list[Resource] = []
    plan_topics = stage3_research.extract_topics(stage2_result)
    only_topics: list[str] | None = None

    total_discovered_count = 0
    total_fetched_count = 0
    all_discovery_errors: list[str] = []

    while loop_count <= max_loops:
        logger.info("stage 3 start")
        _t3 = time.perf_counter()
        await _report_stage(3, "Stage 3: Research Discovery", 3 / 9)
        stage3_result = await stage3_research.run(
            plan_result=stage2_result,
            loop_count=loop_count,
            exclude_urls=exclude_urls,
            only_topics=only_topics,
            provider=provider,
            github_connector=github_connector,
            http_client=http_client,
            tech_stack=request.tech_stack,
            goal=stage1_result.goal,
        )
        total_discovered_count += stage3_result.discovered_count
        total_fetched_count += stage3_result.fetched_count
        all_discovery_errors.extend(stage3_result.errors)
        _record_metrics(stage3_result.metrics)
        logger.info(f"stage 3 done in {time.perf_counter() - _t3:.1f}s")

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
            discovered_count=total_discovered_count,
            fetched_count=total_fetched_count,
            errors=all_discovery_errors,
            loop_count=loop_count,
            metrics=stage3_result.metrics,
        )

        logger.info("stage 4 start")
        _t4 = time.perf_counter()
        await _report_stage(4, "Stage 4: Evidence Review & Validation", 4 / 9)
        evidence_result = await stage4_evidence_validation.run(
            discovery_result=combined_discovery,
            loop_count=loop_count,
            max_retry_loops=max_loops,
            expected_topics=plan_topics,
            goal=stage1_result.goal,
            tech_stack=request.tech_stack,
        )
        _record_metrics(evidence_result.metrics)
        logger.info(f"stage 4 done in {time.perf_counter() - _t4:.1f}s")

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

    # ── Seed Documentation Fallback ───────────────────────────────────────────
    # If fewer than min_sources_per_topic sources are accepted after the loops,
    # fetch the matching seeds through the SSRF-safe fetch_page, pass them through
    # the same relevance gate, and label them origin="curated". If a URL fails, skip it.
    accepted_after_loops = evidence_result.reviewed_resources if evidence_result else []
    seed_shortage = False
    if not accepted_after_loops or len(accepted_after_loops) < cfg.min_sources_per_topic:
        seed_shortage = True
    elif plan_topics:
        for t in plan_topics:
            t_clean = t.strip().lower()
            if sum(1 for r in accepted_after_loops if (r.topic or "").strip().lower() == t_clean) < cfg.min_sources_per_topic:
                seed_shortage = True
                break

    if seed_shortage:
        seed_map = cfg.seed_docs if hasattr(cfg, "seed_docs") else SEED_DOCS
        matching_seed_items: list[tuple[str, str]] = []
        for term in (request.tech_stack or []):
            term_clean = term.strip().lower()
            if term_clean in seed_map:
                matching_seed_items.append((term, seed_map[term_clean]))
            else:
                for k, v in seed_map.items():
                    if k in term_clean or term_clean in k:
                        matching_seed_items.append((term, v))

        seen_seed_urls = {r.url for r in accepted_after_loops if r.url}
        unique_seeds: list[tuple[str, str]] = []
        for term, s_url in matching_seed_items:
            if s_url not in seen_seed_urls:
                seen_seed_urls.add(s_url)
                unique_seeds.append((term, s_url))

        for term, seed_url in unique_seeds:
            total_discovered_count += 1
            try:
                doc = await fetch_page(
                    url=seed_url,
                    resource_type=ResourceType.DOCS,
                    client=http_client,
                )
                total_fetched_count += 1
                seed_res = doc.resource
                seed_res.origin = "curated"
                seed_res.title = doc.resource.title or f"{term} Official Documentation"
                seed_res.excerpt = extract_clean_excerpt(doc.text, max_chars=cfg.max_excerpt_chars)
                seed_res.content_hash = hash_content(doc.text)
                seed_res.last_updated = doc.last_updated
                seed_res.popularity = doc.popularity
                seed_res.topic = plan_topics[0] if plan_topics else "documentation"
                seed_res.reliability = 0.0
                seed_res.freshness_score = 0.0
                seed_res.uncertainty_flag = False

                # Pass through same relevance gate
                is_rel, rel_reason = is_source_relevant_to_goal(
                    seed_res,
                    goal=stage1_result.goal,
                    expected_topics=plan_topics,
                    tech_stack=request.tech_stack,
                )
                if is_rel:
                    scored_doc = score_document(
                        ResearchDocument(
                            resource=seed_res,
                            text=doc.text,
                            last_updated=seed_res.last_updated,
                            popularity=seed_res.popularity,
                        )
                    )
                    scored_res = scored_doc.resource
                    scored_res.origin = "curated"
                    if evidence_result:
                        evidence_result.reviewed_resources.append(scored_res)
                    logger.info("seed_source_accepted", url=seed_url, term=term)
                else:
                    seed_res.is_rejected = True
                    seed_res.rejection_reason = rel_reason
                    if evidence_result:
                        evidence_result.rejected_resources.append(seed_res)
                    logger.info("seed_source_rejected_by_gate", url=seed_url, reason=rel_reason)
            except Exception as exc:
                logger.warning("seed_doc_fetch_failed_skipping", url=seed_url, error=str(exc))
                all_discovery_errors.append(f"Failed to fetch seed doc '{seed_url}': {exc}")

    # ── Stage 5: Explanation & Concept Mapping ────────────────────────────────
    logger.info("stage 5 start")
    _t5 = time.perf_counter()
    await _report_stage(5, "Stage 5: Explanation & Concept Mapping", 5 / 9)
    try:
        stage5_result = await stage5_ai_analysis.run(
            evidence_review=evidence_result,
            plan_result=stage2_result,
            provider=provider,
            auto_repair=True,
            requested_model_id=getattr(request, "model_id", None),
        )
    except Exception as exc:
        logger.warning("stage5_failed_degrading_gracefully", error=str(exc))
        stage5_result = _fallback_explanation_result(stage2_result, evidence_result)

    _record_metrics(stage5_result.metrics)
    logger.info(f"stage 5 done in {time.perf_counter() - _t5:.1f}s")

    # ── Stage 6: Guided Project Work (Optional) ───────────────────────────────
    logger.info("stage 6 start")
    _t6 = time.perf_counter()
    await _report_stage(6, "Stage 6: Guided Project Work", 6 / 9)
    stage6_result: GuidedWorkResult | None = None
    try:
        stage6_result = await stage6_recommendation.run(
            explanation_result=stage5_result,
            plan_result=stage2_result,
        )
        _record_metrics(stage6_result.metrics)
    except Exception as exc:
        logger.warning("stage6_failed_degrading_gracefully", error=str(exc))

    logger.info(f"stage 6 done in {time.perf_counter() - _t6:.1f}s")

    # ── Stage 7: Understanding Check (3 questions across different concepts) ──
    logger.info("stage 7 start")
    _t7 = time.perf_counter()
    await _report_stage(7, "Stage 7: Understanding Check", 7 / 9)
    try:
        stage7_result = await stage7_output_validation.run(
            explanation_result=stage5_result,
            provider=provider,
            project=request.project,
        )
        _record_metrics(stage7_result.metrics)
    except Exception as exc:
        logger.warning("stage7_failed_degrading_gracefully", error=str(exc))
        stage7_result = _fallback_understanding_check_result(stage5_result)
        _record_metrics(stage7_result.metrics)

    logger.info(f"stage 7 done in {time.perf_counter() - _t7:.1f}s")

    # ── Stage 8: Code Assistance Gate ─────────────────────────────────────────
    logger.info("stage 8 start")
    _t8 = time.perf_counter()
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
        _record_metrics(stage8_result.metrics)
    except Exception as exc:
        logger.warning("stage8_failed_degrading_gracefully", error=str(exc))
        stage8_result = CodeGateResult(
            decision=CodeGateDecision(
                level=CodeGateLevel.EXPLAIN,
                reasons=["Default pedagogical policy: explain concepts."],
            )
        )

    logger.info(f"stage 8 done in {time.perf_counter() - _t8:.1f}s")

    # ── Stage 9: Progress & Reflection (Optional) ─────────────────────────────
    logger.info("stage 9 start")
    _t9 = time.perf_counter()
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
        _record_metrics(stage9_result.metrics)
        progress_record = stage9_result.progress
    except Exception as exc:
        logger.warning("stage9_failed_degrading_gracefully", error=str(exc))

    logger.info(f"stage 9 done in {time.perf_counter() - _t9:.1f}s")

    raw_questions = stage7_result.questions if stage7_result else []

    # ── Assemble LearningResponse ─────────────────────────────────────────────
    reviewed_res = evidence_result.reviewed_resources if evidence_result else []

    # Check evidence shortage: if fewer than min_sources_per_topic survive after max loops
    min_per_topic = cfg.min_sources_per_topic
    topics = plan_topics or ([t.title for t in stage2_result.plan.milestones] if stage2_result and stage2_result.plan else [])

    is_shortage = False
    if evidence_result is None or not evidence_result.is_sufficient:
        is_shortage = True
    elif not reviewed_res:
        is_shortage = True
    elif topics:
        for t in topics:
            t_clean = t.strip().lower()
            matching_count = sum(1 for r in reviewed_res if (r.topic or "").strip().lower() == t_clean)
            if matching_count < min_per_topic:
                is_shortage = True
                break
    elif len(reviewed_res) < min_per_topic:
        is_shortage = True

    evidence_status = "insufficient" if is_shortage else "sufficient"

    calibrated_conf = calibrate_confidence_score(
        evidence_result=evidence_result,
        explanation=stage5_result.explanation,
        resources=reviewed_res,
    )
    if is_shortage:
        calibrated_conf = min(0.35, calibrated_conf)

    if stage5_result.explanation:
        stage5_result.explanation.confidence_score = calibrated_conf
        for s in stage5_result.explanation.sections:
            s.content = strip_unbacked_marker(s.content)
            if is_shortage and not s.citations:
                s.support_status = "unsupported"

    warnings: list[ResponseWarning] = []
    if is_shortage:
        warnings.append(
            ResponseWarning(
                severity="critical",
                code="NO_RELIABLE_SOURCES",
                message="Evidence is insufficient to support reliable answers.",
            )
        )
    if stage5_result.explanation and any(
        s.support_status == "unsupported" for s in stage5_result.explanation.sections
    ):
        warnings.append(
            ResponseWarning(
                severity="warning",
                code="UNSUPPORTED_CLAIMS",
                message="One or more explanation sections contain unsupported claims.",
            )
        )
    if all_discovery_errors:
        warnings.append(
            ResponseWarning(
                severity="warning",
                code="PARTIAL_RESULTS",
                message="Results are partial due to search rate limits or discovery errors.",
            )
        )

    # ── Evidence Report Summary ───────────────────────────────────────────────
    rejected_list: list[RejectedSource] = []
    if evidence_result:
        for r in evidence_result.rejected_resources:
            if r.url:
                rejected_list.append(
                    RejectedSource(url=r.url, reason=r.rejection_reason or "rejected")
                )

    evidence_summary = EvidenceSummary(
        discovered=total_discovered_count,
        fetched=total_fetched_count,
        accepted=len(reviewed_res),
        rejected=rejected_list,
        errors=all_discovery_errors,
    )

    # ── Transparency: models_used & FALLBACK_MODEL_USED warning ──────────────
    models_used: list[str] = list(dict.fromkeys(
        m.model_id for m in metrics if m.model_id and isinstance(m.model_id, str)
    ))

    from app.llm.registry import get_model_registry

    reg = get_model_registry()
    all_registered = reg.list_models()
    first_choice_model_id = getattr(request, "model_id", None)
    if not first_choice_model_id and all_registered:
        first_choice_model_id = all_registered[0].id

    fallback_model_used = False
    for m in metrics:
        if m.model_id and isinstance(m.model_id, str) and first_choice_model_id and m.model_id != first_choice_model_id:
            fallback_model_used = True
            break

    if fallback_model_used:
        warnings.append(
            ResponseWarning(
                severity="warning",
                code="FALLBACK_MODEL_USED",
                message=f"One or more stages ran on a fallback model rather than first-choice model '{first_choice_model_id}'.",
            )
        )

    response = LearningResponse(
        learner_profile=stage1_result.profile,
        learning_plan=stage2_result.plan,
        resources=reviewed_res,
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
        confidence_score=calibrated_conf,
        quality_score=0.0,
        evidence_status=evidence_status,
        evidence_summary=evidence_summary,
        warnings=warnings,
        metrics=metrics,
        models_used=models_used,
    )
    response.quality_score = compute_quality_score(response, evidence_status)

    # ── Response Validator & Repair ───────────────────────────────────────────
    report = validate_response(response)
    if not report.is_valid:
        repaired_resp, _ = repair_response(response, report)
        response = repaired_resp
        response.quality_score = compute_quality_score(response, evidence_status)

    # ── Persist to SessionStore ───────────────────────────────────────────────
    store = session_store or get_session_store()
    session = create_session_from_response(
        response,
        questions=raw_questions,
        goal_title=goal_title,
        goal_description=goal_description,
    )
    await store.save(session)

    from app.pipeline.request_cache import save_cached_learning_response

    save_cached_learning_response(request, response)

    return response


async def handle_code_request(
    request: CodeRequest,
    session_store: SessionStore | None = None,
    provider: object | None = None,
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
    async with _get_session_lock(request.response_id):
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
                known_qids = {q.question_id for q in questions_to_eval}
                # Filter out unknown question IDs
                valid_answers = [a for a in request.answers if a.question_id in known_qids]

                eval_results = evaluate_answers(
                    answers=valid_answers,
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
                from app.llm.router import get_provider_for

                provider = await get_provider_for(stage="code")

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

            session.learning_response.full_code = code_to_use
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

