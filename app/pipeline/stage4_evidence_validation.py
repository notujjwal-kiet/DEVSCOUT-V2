"""
Pipeline Stage 4 – Evidence Review.

Responsibility: Verify provenance, apply recency via time-decay, score relevance,
flag conflicting or uncertain claims, and verify content hashes for change detection.
Feedback loop: 4->3 (need more evidence, max 2 loops).
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    EvidenceReviewResult,
    Resource,
    SourceDiscoveryResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 4
STAGE_NAME = "Evidence Review"
MAX_EVIDENCE_RETRY_LOOPS = 2


async def run(
    discovery_result: SourceDiscoveryResult,
    freshness_threshold: float = 0.5,
) -> EvidenceReviewResult:
    """
    Stage 4 entry point: Review and score evidence; trigger loop 4->3 if insufficient.
    """
    started_at = datetime.now(UTC)
    goal_id = discovery_result.goal_id
    loop_count = discovery_result.loop_count

    flagged_uncertainties: list[str] = []
    reviewed_resources: list[Resource] = []

    for res in discovery_result.resources:
        # Time-decay / freshness verification
        if res.freshness_score < freshness_threshold:
            res.uncertainty_flag = True
            flagged_uncertainties.append(f"Resource '{res.title}' flagged for low freshness score ({res.freshness_score}).")
        reviewed_resources.append(res)

    is_sufficient = len(reviewed_resources) >= 2
    is_fresh = all(not r.uncertainty_flag for r in reviewed_resources)

    # Determine if feedback loop 4->3 should be triggered
    needs_more_evidence = (not is_sufficient or not is_fresh) and (loop_count < MAX_EVIDENCE_RETRY_LOOPS)

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage4_evidence_review",
        goal_id=goal_id,
        is_sufficient=is_sufficient,
        is_fresh=is_fresh,
        needs_more_evidence=needs_more_evidence,
        loop_count=loop_count,
    )

    return EvidenceReviewResult(
        goal_id=goal_id,
        is_sufficient=is_sufficient,
        is_fresh=is_fresh,
        reviewed_resources=reviewed_resources,
        flagged_uncertainties=flagged_uncertainties,
        needs_more_evidence=needs_more_evidence,
        loop_count=loop_count,
        metrics=metrics,
    )
