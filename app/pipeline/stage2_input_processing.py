"""
Pipeline Stage 2 – Intent & Learning Plan.

Responsibility: Identify knowledge gaps, targeted outcomes, required depth,
and construct incremental milestones for the learner's project.
Feedback loop: user questions -> 2/3, progress -> 2 (plan update).
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    GoalContextResult,
    IntentPlanResult,
    LearningPlan,
    Milestone,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 2
STAGE_NAME = "Intent & Learning Plan"


async def run(
    goal_context: GoalContextResult,
    depth_level: str = "intermediate",
) -> IntentPlanResult:
    """
    Stage 2 entry point: Generate structured learning plan with milestones.
    """
    started_at = datetime.now(UTC)

    # Stub: Derive knowledge gaps and milestones from goal and stack
    gaps = [
        f"Core principles of {skill}" for skill in goal_context.goal.target_skills
    ] or ["Architecture and best practices"]

    outcomes = [
        f"Master practical implementation of {goal_context.goal.title}",
        "Understand design trade-offs and performance implications",
    ]

    milestones = [
        Milestone(
            title="Foundation & Setup",
            description="Understand core concepts and configure environment.",
            target_outcome="Working baseline setup with no copy-pasted boilerplate.",
            estimated_duration_minutes=30,
        ),
        Milestone(
            title="Core Implementation",
            description="Implement main feature guided by incremental tasks.",
            target_outcome="Core feature functional and understood.",
            estimated_duration_minutes=60,
        ),
    ]

    plan = LearningPlan(
        goal_id=goal_context.goal.goal_id,
        knowledge_gaps=gaps,
        learning_outcomes=outcomes,
        depth_level=depth_level,
        milestones=milestones,
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
        "stage2_learning_plan",
        plan_id=plan.plan_id,
        goal_id=plan.goal_id,
        milestone_count=len(plan.milestones),
    )

    return IntentPlanResult(plan=plan, metrics=metrics)
