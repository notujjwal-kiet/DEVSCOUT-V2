"""
Pipeline Stage 5 – AI Decision Analysis.

Responsibility: Send the validated evidence + processed question to the
local Ollama LLM (via Strands Agents) and obtain a structured analysis
including trade-offs and key decision factors.

TODO (next task): build the Strands agent, prompt templates, and token
tracking integration.
"""
from __future__ import annotations

from datetime import UTC, datetime

import structlog

from app.config import get_settings
from app.models.pipeline import AnalysisResult, StageMetrics, ValidationResult

logger = structlog.get_logger(__name__)

STAGE_ID = 5
STAGE_NAME = "ai_decision_analysis"


async def run(
    validation_result: ValidationResult,
    normalized_question: str,
) -> AnalysisResult:
    """
    Stage 5 entry point.

    Args:
        validation_result:    Output from Stage 4 (validated evidence).
        normalized_question:  Normalised question from Stage 2.

    Returns:
        AnalysisResult – typed model passed to Stage 6.
    """
    cfg = get_settings()
    started_at = datetime.now(UTC)

    # Placeholder: return stub analysis.
    result = AnalysisResult(
        request_id=validation_result.request_id,
        analysis_text="[Placeholder] AI analysis not yet implemented.",
        trade_offs={},
        key_factors=[],
        raw_llm_response=None,
        metrics=StageMetrics(
            stage_id=STAGE_ID,
            stage_name=STAGE_NAME,
            started_at=started_at,
            ended_at=datetime.now(UTC),
        ),
    )

    logger.info(
        "stage5_ai_analysis",
        request_id=result.request_id,
        model=cfg.ollama_model,
        evidence_count=len(validation_result.validated_evidence),
    )
    return result
