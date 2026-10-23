"""
app/api/learn.py

API routes for DEVSCOUT learning sessions:
- POST /api/learn: Initiate background learning session job (202 Accepted).
- GET /api/learn/{job_id}: Query job status and result.
- POST /api/code-request: Evaluate understanding answers / attempt code and request full code.
"""
from __future__ import annotations

import asyncio
import structlog
from fastapi import APIRouter, Depends, HTTPException, status

from app.api.auth import verify_api_key_and_rate_limit
from app.config import get_settings
from app.llm.provider import RateLimitError
from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    CodeRequest,
    LearnerProfile,
    LearnJobRecord,
    LearningPlan,
    LearningResponse,
    LearnRequest,
    ResponseWarning,
)
from app.pipeline.job_store import get_job_store
from app.pipeline.orchestrator import (
    SessionNotFoundError,
    handle_code_request,
    run_learning_pipeline,
)

logger = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key_and_rate_limit)])

_running_tasks: set[asyncio.Task[None]] = set()


async def _run_pipeline_job(job_id: str, request: LearnRequest) -> None:
    """Execute pipeline in background, updating per-stage progress and respecting timeout."""
    cfg = get_settings()
    store = get_job_store()

    async def _on_progress(stage_id: int, stage_name: str, pct: float) -> None:
        await store.update_progress(job_id, stage_id, stage_name, pct)

    if not request.refresh:
        from app.pipeline.request_cache import get_cached_learning_response

        cached = get_cached_learning_response(request)
        if cached is not None:
            logger.info("learn_request_cache_hit", job_id=job_id)
            from app.pipeline.orchestrator import create_session_from_response
            from app.pipeline.session_store import get_session_store

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
            await get_session_store().save(session)
            await store.mark_completed(job_id, cached.to_public())
            return

    try:
        response = await asyncio.wait_for(
            run_learning_pipeline(request, on_progress=_on_progress),
            timeout=cfg.pipeline_timeout_seconds,
        )
        # Strip expected_points before saving to public result
        await store.mark_completed(job_id, response.to_public())
    except TimeoutError:
        logger.warning(
            "learning_pipeline_job_timed_out",
            job_id=job_id,
            timeout=cfg.pipeline_timeout_seconds,
        )
        await store.mark_timed_out(
            job_id,
            f"Pipeline execution timed out after {cfg.pipeline_timeout_seconds} seconds",
        )
    except Exception as exc:
        is_quota = (
            isinstance(exc, RateLimitError)
            or isinstance(getattr(exc, "__cause__", None), RateLimitError)
            or "quota" in str(exc).lower()
            or "rate limit" in str(exc).lower()
            or "resource_exhausted" in str(exc).lower()
            or "429" in str(exc)
        )
        if is_quota:
            friendly_err = "Model quota exceeded. Try again later or check the Gemini plan."
            logger.warning("learning_pipeline_job_quota_exceeded", job_id=job_id, error=friendly_err)
            partial_response = LearningResponse(
                learner_profile=LearnerProfile(),
                learning_plan=LearningPlan(goal_id=request.goal[:50]),
                code_gate_decision=CodeGateDecision(
                    level=CodeGateLevel.EXPLAIN,
                    reasons=["Model quota exceeded."],
                ),
                confidence_score=0.0,
                quality_score=0.0,
                evidence_status="insufficient",
                warnings=[
                    ResponseWarning(
                        severity="warning",
                        code="PARTIAL_RESULTS",
                        message="Results are partial due to model quota exhaustion.",
                    )
                ],
            )
            await store.mark_failed(job_id, friendly_err, result=partial_response.to_public())
        else:
            logger.error("learning_pipeline_job_failed", job_id=job_id, error=str(exc), exc_info=True)
            await store.mark_failed(job_id, "An internal error occurred during pipeline execution.")


@router.post(
    "/learn",
    response_model=LearnJobRecord,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Initiate background learning session",
)
async def create_learning_session(request: LearnRequest) -> LearnJobRecord:
    """
    Initiate background pipeline job for a given developer goal and skill context.
    Returns 202 Accepted with job_id immediately.
    """
    job_store = get_job_store()
    job = await job_store.create_job()

    task = asyncio.create_task(_run_pipeline_job(job.job_id, request))
    _running_tasks.add(task)
    task.add_done_callback(_running_tasks.discard)

    return job


@router.get(
    "/learn/{job_id}",
    response_model=LearnJobRecord,
    summary="Get learning session job status and result",
)
async def get_learning_job(job_id: str) -> LearnJobRecord:
    """Retrieve current status, per-stage progress, and final result for a job."""
    job = await get_job_store().get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found",
        )
    return job


@router.get("/quota", summary="Get Gemini API quota usage and limits")
async def get_quota() -> dict[str, int | None]:
    """Return current daily quota and RPM limits for Gemini API without secrets."""
    from app.llm.rate_gate import get_rate_gate

    return get_rate_gate().get_quota_status()


@router.get("/models", summary="List registered models and quota availability")
async def list_models() -> list[dict[str, object]]:
    """Return registered models with quota availability and limits without secrets."""
    from app.llm.registry import get_model_registry, is_model_available
    from app.llm.rate_gate import get_rate_gate
    from app.providers.registry import get_provider_registry

    reg = get_model_registry()
    gate = get_rate_gate()

    res: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for m in reg.list_models():
        used = gate._get_used_today(m.id) if not m.local else 0
        limit = m.rpd
        available = is_model_available(m)
        seen_ids.add(m.id)

        res.append({
            "id": m.id,
            "label": m.label,
            "provider": m.provider,
            "local": m.local,
            "tier": m.tier,
            "available": available,
            "used_today": used,
            "limit_today": limit,
            "price_in_per_1m": m.price_in_per_1m,
            "price_out_per_1m": m.price_out_per_1m,
        })

    # Include multi-provider registry models (Groq, Cerebras, Ollama)
    try:
        prov_reg = get_provider_registry()
        for info in prov_reg.get_all_models_info():
            if info["id"] not in seen_ids and info.get("provider") != "gemini":
                seen_ids.add(info["id"])
                res.append(info)
    except Exception:
        pass

    return res



@router.post("/code-request", response_model=LearningResponse, summary="Request code or submit answers")
async def request_code(request: CodeRequest) -> LearningResponse:
    """
    Evaluate submitted answers or attempt code against session questions and snippet hashes.
    Unlocks FULL code if learner readiness criteria are satisfied; otherwise returns REQUIRE_CHECK.
    Returns response with expected_points stripped from questions.
    """
    try:
        response = await handle_code_request(request)
        return response.to_public()
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("code_request_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )
