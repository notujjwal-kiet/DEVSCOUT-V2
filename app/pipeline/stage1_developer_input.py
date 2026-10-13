"""
Pipeline Stage 1 – Developer Input.

Responsibility: Accept and lightly validate the raw question from the
frontend, assign a request_id, and pass a DeveloperInput downstream.

TODO (next task): add request deduplication, rate-limiting metadata.
"""
from __future__ import annotations

import structlog

from app.models.pipeline import DeveloperInput

logger = structlog.get_logger(__name__)


async def run(question: str, context: str | None = None) -> DeveloperInput:
    """
    Stage 1 entry point.

    Args:
        question: The raw technical question from the developer.
        context:  Optional additional context (project description, constraints, …).

    Returns:
        DeveloperInput – typed model passed to Stage 2.
    """
    # Placeholder: Pydantic validation is the only logic for now.
    developer_input = DeveloperInput(question=question, context=context)

    logger.info(
        "stage1_developer_input",
        request_id=developer_input.request_id,
        question_len=len(question),
    )
    return developer_input
