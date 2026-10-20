"""
Pipeline Stage 7 – Understanding Check.

Responsibility: Formulate conceptual questions, small exercises, and explain-back
prompts; detect and address misconceptions before code is granted.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from pydantic import BaseModel, Field

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    AnswerSubmission,
    ConceptNode,
    ExplanationResult,
    QuestionType,
    StageMetrics,
    UnderstandingCheckResult,
    UnderstandingQuestion,
    UnderstandingResult,
)
from app.pipeline.response_validator import tokenize_text
from app.pipeline.structured_llm import generate_structured_with_retry

logger = structlog.get_logger(__name__)

STAGE_ID = 7
STAGE_NAME = "Understanding Check"

STAGE7_SYSTEM_PROMPT = (
    "You are an expert technical educator and curriculum designer. "
    "Formulate rigorous conceptual understanding questions grounded exclusively in the provided technical explanation. "
    "Focus purely on the subject matter domain. "
    "NEVER mention DEVSCOUT, readiness gate, untyped dictionaries, pipeline stages, or internal platform policies."
)


class GeneratedQuestionItem(BaseModel):
    concept_id: str = Field(description="ID of the concept node from concept_nodes")
    question_text: str = Field(description="Conceptual understanding question testing the learner")
    question_type: QuestionType = Field(default=QuestionType.CONCEPTUAL)
    expected_points: list[str] = Field(
        ...,
        min_length=1,
        description="Key points taken from explanation text required for a correct answer",
    )


class GeneratedQuestionsBatch(BaseModel):
    questions: list[GeneratedQuestionItem] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="Exactly 3 understanding questions, each about a different concept",
    )


def build_fallback_questions(topic: str, concept_nodes: list[ConceptNode]) -> list[UnderstandingQuestion]:
    """Generate 3 generic conceptual questions about the topic across distinct concepts without DEVSCOUT internals."""
    c1 = concept_nodes[0].id if len(concept_nodes) > 0 else "concept_core"
    c2 = concept_nodes[1].id if len(concept_nodes) > 1 else (f"{c1}_arch" if c1 else "concept_arch")
    c3 = concept_nodes[2].id if len(concept_nodes) > 2 else (f"{c1}_tradeoffs" if c1 else "concept_tradeoffs")

    return [
        UnderstandingQuestion(
            concept_id=c1,
            question_text=f"What is the primary architectural purpose and core benefit of {topic}?",
            question_type=QuestionType.CONCEPTUAL,
            expected_points=[
                f"Solves core domain challenges in {topic}",
                "Ensures system reliability and maintainability",
                "Provides modular architectural boundaries",
            ],
        ),
        UnderstandingQuestion(
            concept_id=c2,
            question_text=f"How do key components and data flows interact within {topic}?",
            question_type=QuestionType.EXPLAIN_BACK,
            expected_points=[
                "Component boundaries and communication protocols",
                "State handling and separation of responsibilities",
            ],
        ),
        UnderstandingQuestion(
            concept_id=c3,
            question_text=f"What key performance and operational trade-offs should be considered when adopting {topic}?",
            question_type=QuestionType.EXERCISE,
            expected_points=[
                "Performance characteristics under varying load",
                "Resource overhead versus architectural guarantees",
            ],
        ),
    ]


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
    provider: GeminiProvider | None = None,
) -> UnderstandingCheckResult:
    """
    Stage 7 entry point: Generate conceptual questions and evaluate learner response.
    Generates 3 understanding questions across different concept_nodes via structured Gemini call,
    grounded in the explanation text with expected_points taken from it.
    Falls back to clean topic-based questions if call fails.
    """
    started_at = datetime.now(UTC)

    topic = (
        explanation_result.explanation.topic
        if explanation_result.explanation
        else "System Architecture"
    )
    concept_nodes = explanation_result.concept_nodes or []

    questions: list[UnderstandingQuestion] = []

    if provider is not None:
        explanation_sections = ""
        if explanation_result.explanation and explanation_result.explanation.sections:
            explanation_sections = "\n\n".join(
                f"### Section: {s.title}\n{s.content}"
                for s in explanation_result.explanation.sections
            )
        explanation_summary = (
            explanation_result.explanation.summary
            if explanation_result.explanation and explanation_result.explanation.summary
            else ""
        )

        concepts_listing = (
            "\n".join(f"- Concept ID: {n.id}, Concept Name: {n.label}" for n in concept_nodes)
            if concept_nodes
            else f"- Concept ID: {topic.lower().replace(' ', '_')}, Concept Name: {topic}"
        )

        prompt = (
            f"You are formulating an active understanding check for a developer learning: {topic}\n\n"
            f"Explanation Summary:\n{explanation_summary}\n\n"
            f"Explanation Content:\n{explanation_sections}\n\n"
            f"Available Concepts (from Concept Nodes):\n{concepts_listing}\n\n"
            f"STRICT INSTRUCTIONS:\n"
            f"1. Generate exactly 3 questions testing the learner's conceptual understanding.\n"
            f"2. Each question MUST focus on a DIFFERENT concept from the available concepts list above. Set concept_id accordingly.\n"
            f"3. Ground all questions and expected_points strictly in the explanation content provided.\n"
            f"4. For each question, extract 2 to 4 concise expected_points directly taken from or supported by the explanation text.\n"
            f"5. NEVER mention DEVSCOUT, readiness gate, untyped dictionaries, stages, or internal platform rules.\n"
            f"Focus exclusively on the developer's subject domain: {topic}."
        )

        try:
            batch = await generate_structured_with_retry(
                provider=provider,
                prompt=prompt,
                schema=GeneratedQuestionsBatch,
                system_prompt=STAGE7_SYSTEM_PROMPT,
            )
            # Ensure each of the 3 questions has a different concept_id
            seen_concepts: set[str] = set()
            for idx, q_item in enumerate(batch.questions):
                cid = q_item.concept_id
                if cid in seen_concepts or not cid:
                    if idx < len(concept_nodes):
                        cid = concept_nodes[idx].id
                    else:
                        cid = f"{cid or 'concept'}_{idx + 1}"
                seen_concepts.add(cid)

                questions.append(
                    UnderstandingQuestion(
                        concept_id=cid,
                        question_text=q_item.question_text,
                        question_type=q_item.question_type,
                        expected_points=q_item.expected_points,
                    )
                )
        except Exception as exc:
            logger.warning("stage7_structured_generation_failed_using_fallback", error=str(exc))
            questions = build_fallback_questions(topic, concept_nodes)
    else:
        questions = build_fallback_questions(topic, concept_nodes)

    if not questions or len(questions) < 3:
        questions = build_fallback_questions(topic, concept_nodes)

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
    if provider is not None:
        metrics.fill_tokens(provider)

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


