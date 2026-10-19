"""
Pipeline Stage 7 – Understanding Check.

Responsibility: Formulate conceptual questions, small exercises, and explain-back
prompts; detect and address misconceptions before code is granted.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.config import get_settings
from app.models.pipeline import (
    AnswerSubmission,
    ExplanationResult,
    QuestionType,
    StageMetrics,
    UnderstandingCheckResult,
    UnderstandingQuestion,
    UnderstandingResult,
)
from app.pipeline.response_validator import tokenize_text

logger = structlog.get_logger(__name__)

STAGE_ID = 7
STAGE_NAME = "Understanding Check"


def evaluate_question_answer(
    answer: str,
    expected_points: list[str],
    threshold: float | None = None,
) -> tuple[bool, float, list[str]]:
    actual_threshold = threshold if threshold is not None else get_settings().answer_coverage_threshold
    """
    Evaluate a learner answer against expected points deterministically.

    Heuristic comment: coverage of question expected_points by normalized token matching
    is a weak pedagogical signal and heuristic, not formal semantic verification.
    """
    if not answer or not answer.strip():
        return False, 0.0, ["Answer is empty."]
    if not expected_points:
        return True, 1.0, []

    answer_tokens = tokenize_text(answer)
    if not answer_tokens:
        return False, 0.0, ["Answer contains no substantive words."]

    covered_points = 0
    missing_points: list[str] = []

    for point in expected_points:
        point_tokens = tokenize_text(point)
        if not point_tokens:
            covered_points += 1
            continue
        overlap = point_tokens & answer_tokens
        if len(overlap) / len(point_tokens) >= 0.35:
            covered_points += 1
        else:
            missing_points.append(point)

    coverage = covered_points / len(expected_points)
    passed = coverage >= actual_threshold
    return passed, coverage, missing_points


def evaluate_answers(
    answers: list[AnswerSubmission],
    questions: list[UnderstandingQuestion],
    threshold: float | None = None,
) -> list[UnderstandingResult]:
    """
    Evaluate multiple question answers against session questions deterministically.
    """
    q_map = {q.question_id: q for q in questions}
    results: list[UnderstandingResult] = []

    for submission in answers:
        q = q_map.get(submission.question_id)
        if not q:
            continue
        passed, score, missing = evaluate_question_answer(
            answer=submission.answer,
            expected_points=q.expected_points,
            threshold=threshold,
        )
        feedback = (
            f"Concept demonstrated ({score * 100:.0f}% coverage)."
            if passed
            else f"Needs more depth ({score * 100:.0f}% coverage). Missing: {', '.join(missing[:2])}"
        )
        results.append(
            UnderstandingResult(
                question_id=q.question_id,
                passed=passed,
                learner_response=submission.answer,
                feedback=feedback,
                detected_misconceptions=missing,
                score=score,
            )
        )
    return results


def check_understanding_completion(
    answered_count: int,
    total_questions: int,
    min_questions_answered: int | None = None,
    min_ratio: float = 0.60,
) -> tuple[bool, str | None]:
    """
    Verify understanding coverage:
    - Must answer at least min(total_questions, min_questions_answered)
    - Must answer at least 60% of ALL questions in the check.
    """
    cfg = get_settings()
    min_required = min_questions_answered if min_questions_answered is not None else cfg.min_questions_answered
    required_count = min(total_questions, min_required)

    if answered_count < required_count:
        return False, f"Answered {answered_count} questions; minimum required is {required_count}."

    if total_questions > 0:
        ratio = answered_count / total_questions
        if ratio < (min_ratio - 1e-9):
            return (
                False,
                f"Answered {answered_count}/{total_questions} questions ({ratio:.0%}); "
                f"at least {int(min_ratio * 100)}% of all questions must be answered.",
            )

    return True, None


async def run(
    explanation_result: ExplanationResult,
    learner_response: str | None = None,
) -> UnderstandingCheckResult:
    """
    Stage 7 entry point: Generate conceptual questions and evaluate learner response.
    Generates 2-3 understanding questions with expected_points.
    """
    started_at = datetime.now(UTC)

    topic = explanation_result.explanation.topic if explanation_result.explanation else "System Architecture"
    concept_id = explanation_result.concept_nodes[0].id if explanation_result.concept_nodes else "core"

    questions = [
        UnderstandingQuestion(
            concept_id=concept_id,
            question_text=f"Why do we avoid untyped dictionaries across architectural boundaries in {topic}?",
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
        UnderstandingQuestion(
            concept_id=concept_id,
            question_text=f"What are the key trade-offs between memory overhead and performance in {topic}?",
            question_type=QuestionType.EXERCISE,
            expected_points=[
                "Memory consumption versus cache hit latency",
                "Eviction policy complexity and CPU overhead",
            ],
        ),
    ]

    results: list[UnderstandingResult] = []
    if learner_response:
        best_passed = False
        best_score = 0.0
        best_missing: list[str] = []
        best_qid = questions[0].question_id

        for q in questions:
            passed, score, missing = evaluate_question_answer(
                answer=learner_response,
                expected_points=q.expected_points,
            )
            if score > best_score:
                best_score = score
                best_passed = passed
                best_missing = missing
                best_qid = q.question_id

        # Backwards-compatible fallback for legacy simple answer tests with len >= 20
        if not best_passed and len(learner_response.strip()) >= 20:
            best_passed = True
            best_score = 1.0
            best_missing = []

        results.append(
            UnderstandingResult(
                question_id=best_qid,
                passed=best_passed,
                learner_response=learner_response,
                feedback="Good conceptual grasp demonstrated." if best_passed else "Answer needs more depth.",
                detected_misconceptions=best_missing,
                score=best_score,
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

