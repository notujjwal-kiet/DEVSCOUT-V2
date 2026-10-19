"""
Pipeline Stage 5 – Explanation & Concept Mapping.

Responsibility: Formulate clear, pedagogical explanations backed by verified
sources (citing Resource IDs or URLs), identify prerequisites, map concept relationships,
and provide concrete real-world engineering examples.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.config import get_settings
from app.llm.provider import GeminiProvider
from app.models.pipeline import (
    EvidenceReviewResult,
    ExplanationResult,
    IntentPlanResult,
    StageMetrics,
)
from app.pipeline.response_validator import repair_explanation, validate_explanation
from app.pipeline.structured_llm import generate_structured_with_retry

logger = structlog.get_logger(__name__)

STAGE_ID = 5
STAGE_NAME = "Explanation & Concept Mapping"

SYSTEM_PROMPT = (
    "You are a principal software engineer and expert mentor. "
    "Formulate rigorous, pedagogical explanations grounded directly in the provided research excerpts. "
    "Paraphrase concepts in your own pedagogical words; do not copy text verbatim. "
    "Every section must cite specific resource_ids from the provided sources in citation_resource_ids. "
    "If any statement or section is not supported by the excerpts, explicitly write "
    "'[Not backed by sources]' in the content. Do not invent or cite unverified sources."
)


def extract_plan_topics(plan_result: IntentPlanResult) -> list[str]:
    """Extract distinct milestone titles or knowledge gaps from the learning plan."""
    topics: list[str] = []
    if plan_result.plan and plan_result.plan.milestones:
        for m in plan_result.plan.milestones:
            t = m.title.strip()
            if t and t not in topics:
                topics.append(t)

    if not topics and plan_result.plan and plan_result.plan.knowledge_gaps:
        for g in plan_result.plan.knowledge_gaps:
            clean_g = g.strip()
            if clean_g and clean_g not in topics:
                topics.append(clean_g)

    if not topics:
        topics.append("Core Architecture and Design")

    return topics


async def run(
    evidence_review: EvidenceReviewResult,
    plan_result: IntentPlanResult,
    provider: GeminiProvider | None = None,
    auto_repair: bool = True,
    top_n: int | None = None,
    top_n_resources: int | None = None,
    excerpt_budget: int | None = None,
) -> ExplanationResult:
    """
    Stage 5 entry point: Generate concept map and explanation with source citations
    via GeminiProvider.generate_structured (no search tool, schema allowed).
    Iterates over milestones/topics from the plan. For each topic, includes top-N
    Resources by evidence_score WITH excerpts in the prompt, respecting excerpt_budget.
    """
    cfg = get_settings()
    effective_top_n = (
        top_n_resources
        if top_n_resources is not None
        else (top_n if top_n is not None else cfg.stage5_top_n_resources)
    )
    effective_budget = (
        excerpt_budget
        if excerpt_budget is not None
        else cfg.stage5_excerpt_budget_chars
    )
    top_n = effective_top_n
    excerpt_budget = effective_budget

    started_at = datetime.now(UTC)

    if provider is None:
        provider = GeminiProvider()

    topics = extract_plan_topics(plan_result)

    # Format available resources with excerpts per topic within excerpt_budget
    remaining_budget = excerpt_budget
    topic_blocks: list[str] = []

    for topic in topics:
        topic_lower = topic.lower()
        matching = [
            r
            for r in evidence_review.reviewed_resources
            if r.topic and (r.topic.lower() in topic_lower or topic_lower in r.topic.lower())
        ]
        other = [r for r in evidence_review.reviewed_resources if r not in matching]

        # Rank resources by evidence_score descending
        matching.sort(key=lambda r: r.evidence_score, reverse=True)
        other.sort(key=lambda r: r.evidence_score, reverse=True)

        selected_resources = (matching + other)[:top_n]

        res_lines: list[str] = []
        for r in selected_resources:
            excerpt_text = ""
            if r.excerpt and remaining_budget > 0:
                allowed = min(len(r.excerpt), remaining_budget)
                excerpt_text = r.excerpt[:allowed]
                remaining_budget -= len(excerpt_text)

            res_lines.append(
                f"- Resource ID: {r.resource_id}\n"
                f"  Title: {r.title}\n"
                f"  URL: {r.url}\n"
                f"  Type: {r.type.value}\n"
                f"  Evidence Score: {r.evidence_score:.2f}\n"
                f"  Excerpt: {excerpt_text or '(No excerpt available)'}"
            )

        topic_content = "\n".join(res_lines) if res_lines else "(No verified sources available for this topic)"
        topic_blocks.append(f"### Milestone / Topic: {topic}\n{topic_content}")

    sources_block = "\n\n".join(topic_blocks)

    overall_topic = " & ".join(topics) if len(topics) <= 2 else f"{topics[0]} and related milestones"

    prompt = (
        f"Generate a comprehensive, evidence-backed explanation and conceptual knowledge map for:\n\n"
        f"Topics to cover: {', '.join(topics)}\n"
        f"Goal ID: {plan_result.plan.goal_id}\n"
        f"Target Outcomes: {', '.join(plan_result.plan.learning_outcomes)}\n"
        f"Knowledge Gaps: {', '.join(plan_result.plan.knowledge_gaps)}\n\n"
        f"Available Verified Resources & Excerpts (from Stage 4 Evidence Review):\n"
        f"{sources_block}\n\n"
        f"STRICT GROUNDING & CITATION RULES:\n"
        f"1. Ground all explanations in the provided excerpts. Paraphrase concepts in your own words; DO NOT copy text verbatim.\n"
        f"2. In explanation.sections, citation_resource_ids MUST ONLY contain Resource IDs from the Available Verified Resources above.\n"
        f"3. DO NOT cite phantom IDs or unverified sources.\n"
        f"4. If a section or claim is not backed by the excerpts, leave citation_resource_ids empty and clearly include '[Not backed by sources]' in the section content.\n"
        f"5. Address every milestone/topic listed above across explanation.sections.\n"
        f"6. Provide concept_nodes and concept_edges representing the domain ontology.\n"
    )

    result = await generate_structured_with_retry(
        provider=provider,
        prompt=prompt,
        schema=ExplanationResult,
        system_prompt=SYSTEM_PROMPT,
    )


    # Validate citations against Stage 4 resources
    report = validate_explanation(result.explanation, evidence_review.reviewed_resources)
    if not report.is_valid and not auto_repair:
        logger.warning(
            "stage5_explanation_validation_issues",
            issues=report.issues,
        )

    if auto_repair:
        repaired_exp, repair_report = repair_explanation(
            result.explanation,
            evidence_review.reviewed_resources,
            report,
        )
        result.explanation = repaired_exp
        if repair_report.repaired:
            logger.info(
                "stage5_explanation_repaired",
                citations_valid=repair_report.citations_valid,
            )

    ended_at = datetime.now(UTC)
    result.metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage5_concept_mapping",
        topic=result.explanation.topic,
        nodes=len(result.concept_nodes),
        sections=len(result.explanation.sections),
    )

    return result

