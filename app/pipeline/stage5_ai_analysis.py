"""
Pipeline Stage 5 – Explanation & Concept Mapping.

Responsibility: Formulate clear, pedagogical explanations backed by verified
sources (citing Resource IDs or URLs), identify prerequisites, map concept relationships,
and provide concrete real-world engineering examples.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

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
    "Formulate rigorous, pedagogical explanations backed strictly by verified research sources. "
    "Every section must cite specific resource IDs or URLs from the provided sources. "
    "If any statement or section is not supported by the provided sources, explicitly write "
    "'[Not backed by sources]' in the content. Do not invent or cite unverified sources."
)


async def run(
    evidence_review: EvidenceReviewResult,
    plan_result: IntentPlanResult,
    provider: GeminiProvider | None = None,
    auto_repair: bool = True,
) -> ExplanationResult:
    """
    Stage 5 entry point: Generate concept map and explanation with source citations
    via GeminiProvider.generate_structured (no search tool, schema allowed).
    Validates that citations reference only verified Stage 4 resources.
    Flags or repairs uncited and phantom-cited claims.
    """
    started_at = datetime.now(UTC)

    if provider is None:
        provider = GeminiProvider()

    # Format available resources from Stage 4
    sources_text_lines = []
    for r in evidence_review.reviewed_resources:
        sources_text_lines.append(
            f"- Resource ID: {r.resource_id} | Type: {r.type.value} | URL: {r.url} | Title: {r.title}"
        )
    sources_block = (
        "\n".join(sources_text_lines)
        if sources_text_lines
        else "(No verified sources available)"
    )

    topic = (
        plan_result.plan.knowledge_gaps[0]
        if plan_result.plan.knowledge_gaps
        else "Core Architecture and Design"
    )

    prompt = (
        f"Generate a comprehensive, evidence-backed explanation and conceptual knowledge map for:\n\n"
        f"Topic: {topic}\n"
        f"Goal ID: {plan_result.plan.goal_id}\n"
        f"Target Outcomes: {', '.join(plan_result.plan.learning_outcomes)}\n"
        f"Knowledge Gaps: {', '.join(plan_result.plan.knowledge_gaps)}\n\n"
        f"Available Verified Resources (from Stage 4 Evidence Review):\n"
        f"{sources_block}\n\n"
        f"STRICT CITATION RULES:\n"
        f"1. In explanation.sections, citation_resource_ids MUST ONLY contain Resource IDs or URLs from the Available Verified Resources list above.\n"
        f"2. DO NOT cite phantom IDs or external URLs not in the list.\n"
        f"3. If a section or claim is not backed by the sources above, leave citation_resource_ids empty and clearly include '[Not backed by sources]' in the section content.\n"
        f"4. Provide concept_nodes and concept_edges representing the domain ontology.\n"
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

