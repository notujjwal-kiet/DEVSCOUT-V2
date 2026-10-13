"""
Pipeline Stage 8 – Decision Report.

Responsibility: Assemble the final DecisionReport from all upstream
stage outputs and return it to the API layer for delivery to the frontend.

TODO (next task): persist report to PostgreSQL, attach full pipeline
metrics timeline.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.models.pipeline import (
    DecisionReport,
    DeveloperInput,
    OutputValidationResult,
    ProcessedInput,
    RecommendationResult,
    ResearchResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 8
STAGE_NAME = "decision_report"


async def run(
    developer_input: DeveloperInput,
    processed_input: ProcessedInput,
    research_result: ResearchResult,
    recommendation_result: RecommendationResult,
    output_validation: OutputValidationResult,
    all_metrics: list[StageMetrics],
) -> DecisionReport:
    """
    Stage 8 entry point.

    Args:
        developer_input:       Stage 1 output.
        processed_input:       Stage 2 output.
        research_result:       Stage 3 output (latest iteration).
        recommendation_result: Stage 6 output.
        output_validation:     Stage 7 output.
        all_metrics:           Collected StageMetrics from every stage.

    Returns:
        DecisionReport – the final artifact sent to the frontend.
    """
    started_at = datetime.now(UTC)

    report = DecisionReport(
        request_id=developer_input.request_id,
        question=developer_input.question,
        summary="[Placeholder] Report summary not yet generated.",
        recommendations=recommendation_result.recommendations,
        overall_confidence=recommendation_result.overall_confidence,
        evidence_items=research_result.evidence_items,
        pipeline_metrics=all_metrics,
    )

    all_metrics.append(
        StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        )
    )

    logger.info(
        "stage8_decision_report",
        report_id=report.report_id,
        request_id=report.request_id,
        confidence=report.overall_confidence,
    )
    return report
