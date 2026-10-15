"""
Pipeline Stage 6 – Guided Project Work.

Responsibility: Split the project into concrete, sequential steps with
task-by-task guidance and debugging hints, leaving implementation to the learner.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    ExplanationResult,
    GuidedWorkResult,
    GuidedWorkStep,
    IntentPlanResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 6
STAGE_NAME = "Guided Project Work"


async def run(
    explanation_result: ExplanationResult,
    plan_result: IntentPlanResult,
) -> GuidedWorkResult:
    """
    Stage 6 entry point: Break project into actionable tasks for learner to implement.
    """
    started_at = datetime.now(UTC)
    goal_id = plan_result.plan.goal_id

    steps = [
        GuidedWorkStep(
            step_number=1,
            title="Define Domain Schema",
            guidance="Define strict domain models representing your entities using Pydantic or dataclasses.",
            hints=[
                "Avoid untyped dictionaries.",
                "Ensure validation is enforced at system boundaries.",
            ],
            learner_task="Write the schema module and unit tests validating edge conditions.",
        ),
        GuidedWorkStep(
            step_number=2,
            title="Implement Core Logic & State Flow",
            guidance="Build the primary processing logic step-by-step.",
            hints=[
                "Keep methods focused and deterministic where possible.",
                "Verify return types match downstream stage expectations.",
            ],
            learner_task="Implement the core processing class and verify it with automated tests.",
        ),
    ]

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage6_guided_work",
        goal_id=goal_id,
        step_count=len(steps),
    )

    return GuidedWorkResult(
        goal_id=goal_id,
        steps=steps,
        metrics=metrics,
    )
