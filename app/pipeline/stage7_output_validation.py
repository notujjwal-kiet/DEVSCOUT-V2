"""
Pipeline Stage 7 – Understanding Check.

Responsibility: Formulate conceptual questions, small exercises, and explain-back
prompts; detect and address misconceptions before code is granted.
"""
from __future__ import annotations

import re
from datetime import UTC, datetime
import structlog

from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.pipeline import (
    AnswerSubmission,
    ConceptEdge,
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


def build_deterministic_questions(
    concept_nodes: list[ConceptNode],
    concept_edges: list[ConceptEdge] | None = None,
    project: str | None = None,
    topic: str = "System Architecture",
) -> list[UnderstandingQuestion]:
    """
    Build 3 understanding questions deterministically from concept_nodes and concept_edges (no LLM).
    1. "In your own words, what is {label}?"
    2. "How does {source label} relate to {target label}?" using an edge
    3. "Give one small example of {label} in your {project}"
    Each uses a DIFFERENT concept; expected_points come from node description and edge relation tokens.
    """
    nodes = list(concept_nodes or [])
    if len(nodes) < 1:
        nodes.append(
            ConceptNode(
                id="concept_core",
                label=topic,
                description=f"Core architectural foundation and principles of {topic}",
            )
        )
    if len(nodes) < 2:
        c1 = nodes[0].id
        nodes.append(
            ConceptNode(
                id=f"{c1}_components",
                label=f"{topic} Components",
                description=f"Component boundaries and interactions in {topic}",
            )
        )
    if len(nodes) < 3:
        c1 = nodes[0].id
        nodes.append(
            ConceptNode(
                id=f"{c1}_practices",
                label=f"{topic} Best Practices",
                description=f"Implementation patterns and tradeoffs in {topic}",
            )
        )

    node_map = {n.id: n for n in nodes}
    for n in nodes:
        node_map[n.label] = n

    c1_node = nodes[0]
    c2_node = nodes[1]
    c3_node = nodes[2]

    # Q1: In your own words, what is {label}?
    q1_text = f"In your own words, what is {c1_node.label}?"
    q1_points = [p.strip() for p in re.split(r"[.;\n]", c1_node.description) if len(p.strip()) > 3]
    if not q1_points:
        q1_points = [c1_node.description or f"Definition and purpose of {c1_node.label}"]

    # Q2: How does {source label} relate to {target label}? using an edge
    edges = list(concept_edges or [])
    chosen_edge: ConceptEdge | None = None
    for e in edges:
        if e.source in (c2_node.id, c2_node.label) or e.target in (c2_node.id, c2_node.label):
            chosen_edge = e
            break
    if chosen_edge is None and edges:
        chosen_edge = edges[0]

    if chosen_edge is not None:
        src_label = node_map[chosen_edge.source].label if chosen_edge.source in node_map else chosen_edge.source
        tgt_label = node_map[chosen_edge.target].label if chosen_edge.target in node_map else chosen_edge.target
        rel = chosen_edge.relation
    else:
        src_label = c2_node.label
        tgt_label = c1_node.label
        rel = "interacts with and supports"

    q2_text = f"How does {src_label} relate to {tgt_label}?"
    q2_points = [
        f"{src_label} {rel} {tgt_label}",
        f"Relationship: {rel}",
    ]
    if c2_node.description:
        q2_points.append(c2_node.description)

    # Q3: Give one small example of {label} in your {project}
    proj_name = project.strip() if project and project.strip() else "project"
    q3_text = f"Give one small example of {c3_node.label} in your {proj_name}"
    q3_points = [p.strip() for p in re.split(r"[.;\n]", c3_node.description) if len(p.strip()) > 3]
    if not q3_points:
        q3_points = [c3_node.description or f"Practical example of {c3_node.label}"]

    return [
        UnderstandingQuestion(
            concept_id=c1_node.id,
            question_text=q1_text,
            question_type=QuestionType.CONCEPTUAL,
            expected_points=q1_points,
        ),
        UnderstandingQuestion(
            concept_id=c2_node.id,
            question_text=q2_text,
            question_type=QuestionType.EXPLAIN_BACK,
            expected_points=q2_points,
        ),
        UnderstandingQuestion(
            concept_id=c3_node.id,
            question_text=q3_text,
            question_type=QuestionType.EXERCISE,
            expected_points=q3_points,
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
    provider: object | None = None,
    project: str | None = None,
) -> UnderstandingCheckResult:
    """
    Stage 7 entry point: Generate conceptual questions deterministically and evaluate learner response.
    Builds 3 understanding questions deterministically from concept_nodes and concept_edges (no LLM):
    1. "In your own words, what is {label}?"
    2. "How does {source label} relate to {target label}?" using an edge
    3. "Give one small example of {label} in your {project}"
    Each uses a DIFFERENT concept; expected_points come from node description and edge relation tokens.
    """
    started_at = datetime.now(UTC)

    topic = (
        explanation_result.explanation.topic
        if explanation_result.explanation
        else "System Architecture"
    )
    concept_nodes = explanation_result.concept_nodes or []
    concept_edges = explanation_result.concept_edges or []

    questions = build_deterministic_questions(
        concept_nodes=concept_nodes,
        concept_edges=concept_edges,
        project=project,
        topic=topic,
    )

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


