"""
Pipeline Stage 1 – Developer Goal & Skill Context.

Responsibility: Capture what the developer wants to build and learn,
including experience level, tech stack, constraints, and current project context.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    ExperienceLevel,
    GoalContextResult,
    LearnerProfile,
    LearningGoal,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 1
STAGE_NAME = "Developer Goal & Skill Context"


async def run(
    goal_title: str,
    goal_description: str,
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER,
    tech_stack: list[str] | None = None,
    constraints: list[str] | None = None,
    current_project: str | None = None,
    target_skills: list[str] | None = None,
) -> GoalContextResult:
    """
    Stage 1 entry point: Establish learner profile and learning goal context.
    """
    started_at = datetime.now(UTC)

    profile = LearnerProfile(
        experience_level=experience_level,
        tech_stack=tech_stack or [],
        constraints=constraints or [],
        current_project=current_project,
    )

    goal = LearningGoal(
        title=goal_title,
        description=goal_description,
        target_skills=target_skills or [],
        project_context=current_project,
        constraints=constraints or [],
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
        "stage1_goal_context",
        learner_id=profile.learner_id,
        goal_id=goal.goal_id,
        experience_level=profile.experience_level,
    )

    return GoalContextResult(profile=profile, goal=goal, metrics=metrics)
