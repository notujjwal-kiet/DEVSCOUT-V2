"""
Pipeline Stage 9 – Progress & Reflection.

Responsibility: Record learning history, completed topics, learner feedback,
reflections, and recommend next steps; persist as updated learning context.
Feedback loop: progress -> 2 (plan update).
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    ProgressRecord,
    ProgressReflectionResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 9
STAGE_NAME = "Progress & Reflection"


async def run(
    learner_id: str,
    goal_id: str,
    completed_topics: list[str] | None = None,
    reflection_notes: str | None = None,
    next_steps: list[str] | None = None,
) -> ProgressReflectionResult:
    """
    Stage 9 entry point: Persist progress record and reflections.
    """
    started_at = datetime.now(UTC)

    completed = completed_topics or []
    recommended_next = next_steps or [
        "Review code gate feedback",
        "Implement independent exercise without assistance",
    ]

    progress = ProgressRecord(
        learner_id=learner_id,
        goal_id=goal_id,
        completed_topics=completed,
        reflection_notes=reflection_notes,
        next_recommended_steps=recommended_next,
    )

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage9_progress_reflection",
        record_id=progress.record_id,
        learner_id=learner_id,
        completed_count=len(completed),
    )

    return ProgressReflectionResult(
        progress=progress,
        metrics=metrics,
    )
