"""
Pipeline Stage 2 – Input Processing.

Responsibility: Normalise the question, extract named entities (tech
stack references, decision entities), and infer the decision intent.

TODO (next task): implement NLP entity extraction via the Strands agent.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.models.pipeline import DeveloperInput, ProcessedInput, StageMetrics

logger = structlog.get_logger(__name__)

STAGE_ID = 2
STAGE_NAME = "input_processing"


async def run(developer_input: DeveloperInput) -> ProcessedInput:
    """
    Stage 2 entry point.

    Args:
        developer_input: Output from Stage 1.

    Returns:
        ProcessedInput – typed model passed to Stage 3.
    """
    started_at = datetime.now(UTC)

    # Placeholder: pass through with minimal normalisation.
    normalized = developer_input.question.strip()

    result = ProcessedInput(
        request_id=developer_input.request_id,
        original_question=developer_input.question,
        normalized_question=normalized,
        entities=[],       # TODO: populate via NLP
        intent=None,       # TODO: classify intent
        context=developer_input.context,
        metrics=StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        ),
    )

    logger.info(
        "stage2_input_processing",
        request_id=result.request_id,
        entities=result.entities,
    )
    return result
