"""
Pipeline Stage 7 – Understanding Check.

Responsibility: Formulate conceptual questions, small exercises, and explain-back
prompts; detect and address misconceptions before code is granted.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    ExplanationResult,
    QuestionType,
    StageMetrics,
    UnderstandingCheckResult,
    UnderstandingQuestion,
    UnderstandingResult,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 7
STAGE_NAME = "Understanding Check"


async def run(
    explanation_result: ExplanationResult,
    learner_response: str | None = None,
) -> UnderstandingCheckResult:
    """
    Stage 7 entry point: Generate conceptual questions and evaluate learner response.
    """
    started_at = datetime.now(UTC)

    concept_id = explanation_result.concept_nodes[0].id if explanation_result.concept_nodes else "core"

    questions = [
        UnderstandingQuestion(
            concept_id=concept_id,
            question_text="Why do we avoid untyped dictionaries across architectural boundaries?",
            question_type=QuestionType.CONCEPTUAL,
            expected_points=[
                "Type safety and static checking",
                "Schema validation at boundaries",
                "Self-documenting domain models",
            ],
        ),
        UnderstandingQuestion(
            concept_id=concept_id,
            question_text="Explain in your own words how the readiness gate prevents passive copy-pasting.",
            question_type=QuestionType.EXPLAIN_BACK,
            expected_points=[
                "Forces active recall or implementation attempt first",
                "Verifies conceptual understanding before code assistance",
            ],
        ),
    ]

    results: list[UnderstandingResult] = []
    if learner_response:
        # Stub check evaluating learner's answer
        passed = len(learner_response.strip()) >= 20
        results.append(
            UnderstandingResult(
                question_id=questions[0].question_id,
                passed=passed,
                learner_response=learner_response,
                feedback="Good conceptual grasp demonstrated." if passed else "Answer needs more depth.",
                detected_misconceptions=[] if passed else ["Incomplete understanding of boundary validation"],
                score=1.0 if passed else 0.5,
            )
        )

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage7_understanding_check",
        question_count=len(questions),
        evaluated_count=len(results),
    )

    return UnderstandingCheckResult(
        questions=questions,
        results=results,
        metrics=metrics,
    )
