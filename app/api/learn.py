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
    Returns complete validated LearningResponse with expected_points stripped from questions.
    """
    try:
        response = await run_learning_pipeline(request)
        # Req 3: never return expected_points to clients
        return response.to_public()
    except Exception as exc:
        # Req 5: generic 500 message; log details internally only
        logger.error("learn_pipeline_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )


@router.post("/code-request", response_model=LearningResponse, summary="Request code or submit answers")
async def request_code(request: CodeRequest) -> LearningResponse:
    """
    Evaluate submitted answers or attempt code against session questions and snippet hashes.
    Unlocks FULL code if learner readiness criteria are satisfied; otherwise returns REQUIRE_CHECK.
    Returns response with expected_points stripped from questions.
    """
    try:
        response = await handle_code_request(request)
        # Req 3: never return expected_points to clients
        return response.to_public()
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )
    except Exception as exc:
        # Req 5: generic 500 message; log details internally only
        logger.error("code_request_failed", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again later.",
        )
