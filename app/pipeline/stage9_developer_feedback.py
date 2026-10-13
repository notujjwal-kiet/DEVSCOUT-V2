"""
Pipeline Stage 9 – Developer Feedback.

Responsibility: Accept post-report feedback from the developer, store it
in PostgreSQL, and feed aggregated signals into the Context & Memory layer
to improve future recommendations.

TODO (next task): persist feedback to the DB, compute aggregated signal
scores, and update the context memory embeddings.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.models.pipeline import DeveloperFeedback, FeedbackAck, StageMetrics

logger = structlog.get_logger(__name__)

STAGE_ID = 9
STAGE_NAME = "developer_feedback"


async def run(feedback: DeveloperFeedback) -> FeedbackAck:
    """
    Stage 9 entry point.

    Args:
        feedback: Feedback submitted by the developer via the API.

    Returns:
        FeedbackAck – confirmation that feedback was received/stored.
    """
    started_at = datetime.now(UTC)

    # Placeholder: acknowledge without persisting.
    ack = FeedbackAck(
        feedback_id=feedback.feedback_id,
        stored=False,   # TODO: set True after DB write
        message="[Placeholder] Feedback received but not yet persisted.",
    )

    _metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=datetime.now(UTC),
    )

    logger.info(
        "stage9_developer_feedback",
        feedback_id=ack.feedback_id,
        report_id=feedback.report_id,
        rating=feedback.rating,
    )
    return ack
