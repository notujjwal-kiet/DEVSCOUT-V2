"""
Pipeline Stage 7 – Output Validation.

Responsibility: Validate the recommendation result against the output
schema, verify all citations reference real evidence IDs, and ensure the
overall confidence meets the configured threshold.  Trigger
repair/retry if validation fails.

TODO (next task): implement citation cross-referencing and auto-repair
prompt for low-confidence or schema-invalid outputs.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.config import get_settings
from app.models.pipeline import (
    OutputValidationResult,
    RecommendationResult,
    StageMetrics,
    ValidationResult,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 7
STAGE_NAME = "output_validation"


async def run(
    recommendation_result: RecommendationResult,
    validation_result: ValidationResult,
) -> OutputValidationResult:
    """
    Stage 7 entry point.

    Args:
        recommendation_result: Output from Stage 6.
        validation_result:     Stage 4 output – used for citation checking.

    Returns:
        OutputValidationResult – typed model passed to Stage 8.
    """
    cfg = get_settings()
    started_at = datetime.now(UTC)
    issues: list[str] = []

    # Placeholder: basic confidence check only.
    confidence_ok = (
        recommendation_result.overall_confidence >= cfg.confidence_threshold
    )
    if not confidence_ok:
        issues.append(
            f"Overall confidence {recommendation_result.overall_confidence:.2f} "
            f"is below threshold {cfg.confidence_threshold:.2f}."
        )

    result = OutputValidationResult(
        request_id=recommendation_result.request_id,
        is_valid=confidence_ok and len(issues) == 0,
        schema_valid=True,           # Pydantic already enforced this
        citations_valid=True,        # TODO: cross-reference citation IDs
        confidence_acceptable=confidence_ok,
        repair_attempted=False,
        issues=issues,
        metrics=StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        ),
    )

    logger.info(
        "stage7_output_validation",
        request_id=result.request_id,
        is_valid=result.is_valid,
        issues=result.issues,
    )
    return result
