"""
Pipeline Stage 4 – Source Freshness & Evidence Validation.

Responsibility: Validate that collected evidence is recent enough and
sufficient in quantity.  If not, signal that Stage 3 must be re-run
(up to MAX_EVIDENCE_RETRY_LOOPS times).

TODO (next task): implement freshness heuristics (publication date checks,
domain trust scoring, minimum evidence count threshold).
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.config import get_settings
from app.models.pipeline import ResearchResult, StageMetrics, ValidationResult

logger = structlog.get_logger(__name__)

STAGE_ID = 4
STAGE_NAME = "source_freshness_evidence_validation"


async def run(research_result: ResearchResult) -> ValidationResult:
    """
    Stage 4 entry point.

    Args:
        research_result: Output from Stage 3.

    Returns:
        ValidationResult.  If ``is_sufficient`` is False and
        ``loop_count`` is below the configured maximum, the orchestrator
        should call Stage 3 again and feed its output back here.
    """
    cfg = get_settings()
    started_at = datetime.now(UTC)
    issues: list[str] = []

    # Placeholder: treat any non-empty evidence list as sufficient.
    has_evidence = len(research_result.evidence_items) > 0
    if not has_evidence:
        issues.append("No evidence items collected.")

    result = ValidationResult(
        request_id=research_result.request_id,
        is_sufficient=has_evidence,
        is_fresh=has_evidence,
        issues=issues,
        loop_count=research_result.loop_count,
        validated_evidence=research_result.evidence_items,
        metrics=StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        ),
    )

    logger.info(
        "stage4_validation",
        request_id=result.request_id,
        is_sufficient=result.is_sufficient,
        loop=result.loop_count,
        max_loops=cfg.max_evidence_retry_loops,
    )
    return result
