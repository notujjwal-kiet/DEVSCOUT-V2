"""
app/api/learn.py

API routes for DEVSCOUT learning sessions:
- POST /api/learn: Initiate an end-to-end learning session.
- POST /api/code-request: Evaluate understanding answers / attempt code and request full code.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException

from app.models.pipeline import CodeRequest, LearningResponse, LearnRequest
from app.pipeline.orchestrator import (
    SessionNotFoundError,
    handle_code_request,
    run_learning_pipeline,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/learn", response_model=LearningResponse, summary="Initiate learning session")
async def create_learning_session(request: LearnRequest) -> LearningResponse:
    """
    Run pipeline end-to-end for a given developer goal, skill context, and stack.
    Returns complete validated LearningResponse.
    """
    try:
        response = await run_learning_pipeline(request)
        return response
    except Exception as exc:
        logger.error("learn_pipeline_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to execute learning pipeline: {exc}",
        )


@router.post("/code-request", response_model=LearningResponse, summary="Request code or submit answers")
async def request_code(request: CodeRequest) -> LearningResponse:
    """
    Evaluate submitted answers or attempt code against session questions and snippet hashes.
    Unlocks FULL code if learner readiness criteria are satisfied; otherwise returns REQUIRE_CHECK.
    """
    try:
        response = await handle_code_request(request)
        return response
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("code_request_failed", error=str(exc))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process code request: {exc}",
        )
