"""
Pipeline Stage 2 – Intent & Learning Plan.

Responsibility: Identify knowledge gaps, targeted outcomes, required depth,
and construct incremental milestones for the learner's project.
Feedback loop: user questions -> 2/3, progress -> 2 (plan update).
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    GoalContextResult,
    IntentPlanResult,
    LearningPlan,
    StageMetrics,
)
from app.pipeline.structured_llm import generate_structured_with_retry

logger = structlog.get_logger(__name__)

STAGE_ID = 2
STAGE_NAME = "Intent & Learning Plan"

SYSTEM_PROMPT = (
    "You are an expert pedagogical curriculum designer and technical mentor. "
    "Break down technical learning goals into structured, evidence-oriented learning plans. "
    "Identify specific knowledge gaps, clear learning outcomes, and practical incremental milestones."
)


async def run(
    goal_context: GoalContextResult,
    depth_level: str = "intermediate",
    provider: GeminiProvider | None = None,
) -> IntentPlanResult:
    """
    Stage 2 entry point: Generate structured learning plan with milestones
    via GeminiProvider.generate_structured (no search tool, schema allowed).
    """
    started_at = datetime.now(UTC)

    if provider is None:
        provider = GeminiProvider()

    prompt = (
        f"Design a structured learning plan for the following learner and goal:\n\n"
        f"Learner Profile:\n"
        f"- Experience Level: {goal_context.profile.experience_level.value}\n"
        f"- Tech Stack: {', '.join(goal_context.profile.tech_stack) if goal_context.profile.tech_stack else 'General'}\n"
        f"- Constraints: {', '.join(goal_context.profile.constraints) if goal_context.profile.constraints else 'None'}\n\n"
        f"Learning Goal:\n"
        f"- Title: {goal_context.goal.title}\n"
        f"- Description: {goal_context.goal.description}\n"
        f"- Target Skills: {', '.join(goal_context.goal.target_skills) if goal_context.goal.target_skills else 'Software Engineering'}\n"
        f"- Goal ID: {goal_context.goal.goal_id}\n"
        f"- Required Depth Level: {depth_level}\n\n"
        f"Requirements:\n"
        f"- goal_id must be exactly '{goal_context.goal.goal_id}'\n"
        f"- knowledge_gaps: list key conceptual, architectural, or design gaps\n"
        f"- learning_outcomes: concrete measurable skills the developer will acquire\n"
        f"- depth_level: '{depth_level}'\n"
        f"- milestones: 2 to 4 incremental milestones with title, description, target_outcome, and estimated_duration_minutes\n"
    )

    plan = await generate_structured_with_retry(
        provider=provider,
        prompt=prompt,
        schema=LearningPlan,
        system_prompt=SYSTEM_PROMPT,
    )

    import uuid

    # Re-enforce deterministic pipeline associations
    plan.goal_id = goal_context.goal.goal_id
    plan.depth_level = depth_level
    plan.plan_id = str(uuid.uuid4())
    plan.created_at = datetime.now(UTC)

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )
    if provider is not None:
        metrics.fill_tokens(provider)


    logger.info(
        "stage2_learning_plan",
        plan_id=plan.plan_id,
        goal_id=plan.goal_id,
        milestone_count=len(plan.milestones),
    )

    return IntentPlanResult(plan=plan, metrics=metrics)

