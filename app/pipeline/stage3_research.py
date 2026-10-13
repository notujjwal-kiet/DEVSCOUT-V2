"""
Pipeline Stage 3 – Research & Evidence Collection.

Responsibility: Use web research (live or cached) and the PostgreSQL
evidence store to gather sources relevant to the processed question.

TODO (next task): implement live web search + pgvector similarity recall.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.config import get_settings
from app.models.pipeline import (
    EvidenceItem,
    ProcessedInput,
    ResearchMode,
    ResearchResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 3
STAGE_NAME = "research_evidence_collection"


async def run(
    processed_input: ProcessedInput,
    loop_count: int = 0,
) -> ResearchResult:
    """
    Stage 3 entry point.

    Args:
        processed_input: Output from Stage 2 (or Stage 4 on retry).
        loop_count:       Current retry loop iteration (max set in config).

    Returns:
        ResearchResult – typed model passed to Stage 4.
    """
    cfg = get_settings()
    started_at = datetime.now(UTC)
    mode = ResearchMode(cfg.web_research_mode)

    # Placeholder: return empty evidence list.
    evidence_items: list[EvidenceItem] = []

    result = ResearchResult(
        request_id=processed_input.request_id,
        evidence_items=evidence_items,
        research_mode=mode,
        loop_count=loop_count,
        metrics=StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        ),
    )

    logger.info(
        "stage3_research",
        request_id=result.request_id,
        evidence_count=len(evidence_items),
        loop=loop_count,
        mode=mode,
    )
    return result
