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
from app.llm.provider import LLMProvider
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
    "Every section must include quote-anchored citations: each citation carries resource_id and "
    "supporting_quote (a verbatim quote of at most 15 words from the cited source text). "
    "NEVER discuss missing sources or lack of excerpts inside summary or section content. "
    "DO NOT include '[Not backed by sources]' in section content; support status is tracked externally. "
    "Do not invent quotes or cite unverified sources."
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
    provider: LLMProvider | object | None = None,
    auto_repair: bool = True,
    top_n: int | None = None,
    top_n_resources: int | None = None,
    excerpt_budget: int | None = None,
    requested_model_id: str | None = None,
) -> ExplanationResult:
    """
    Stage 5 entry point: Generate concept map and explanation with source citations
    via routed provider generate_structured (no search tool, schema allowed).
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
        from app.llm.router import get_provider_for

        provider = await get_provider_for(stage=STAGE_ID, requested_id=requested_model_id)

    topics = extract_plan_topics(plan_result)

    # Cap prompt size: send at most 6 sources total, and limit excerpt chars per source
    MAX_STAGE5_SOURCES = 6
    max_excerpt_per_source = cfg.max_excerpt_chars
    remaining_budget = excerpt_budget

    candidate_resources: list[Resource] = []
    seen_ids: set[str] = set()
    for topic in topics:
        topic_lower = topic.lower()
        matching = [
            r
            for r in evidence_review.reviewed_resources
            if r.topic and (r.topic.lower() in topic_lower or topic_lower in r.topic.lower())
        ]
        other = [r for r in evidence_review.reviewed_resources if r not in matching]
        matching.sort(key=lambda r: r.evidence_score, reverse=True)
        other.sort(key=lambda r: r.evidence_score, reverse=True)
        for r in (matching + other):
            if r.resource_id not in seen_ids:
                seen_ids.add(r.resource_id)
                candidate_resources.append(r)
                if len(candidate_resources) >= MAX_STAGE5_SOURCES:
                    break
        if len(candidate_resources) >= MAX_STAGE5_SOURCES:
            break

    if len(candidate_resources) < MAX_STAGE5_SOURCES:
        remaining_pool = [r for r in evidence_review.reviewed_resources if r.resource_id not in seen_ids]
        remaining_pool.sort(key=lambda r: r.evidence_score, reverse=True)
        for r in remaining_pool[: MAX_STAGE5_SOURCES - len(candidate_resources)]:
            candidate_resources.append(r)
            seen_ids.add(r.resource_id)

    topic_blocks: list[str] = []
    included_resource_ids: set[str] = set()

    for topic in topics:
        topic_lower = topic.lower()
        matching = [
            r
            for r in candidate_resources
            if r.topic and (r.topic.lower() in topic_lower or topic_lower in r.topic.lower())
        ]
        other = [r for r in candidate_resources if r not in matching]
        selected_resources = (matching + other)[:top_n]

        res_lines: list[str] = []
        for r in selected_resources:
            included_resource_ids.add(r.resource_id)
            excerpt_text = ""
            if r.excerpt and remaining_budget > 0:
                allowed_per_source = min(len(r.excerpt), max_excerpt_per_source)
                allowed = min(allowed_per_source, remaining_budget)
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
        f"2. In explanation.sections, each citation in citations MUST carry resource_id (from the Available Verified Resources above) "
        f"and supporting_quote (a verbatim quote of max 15 words from the cited resource's text).\n"
        f"3. DO NOT cite phantom IDs, unverified sources, or invent quotes.\n"
        f"4. If a section or claim is not backed by the excerpts, leave citations empty. "
        f"DO NOT mention missing sources, lack of excerpts, or unverified documentation inside summary or section content. "
        f"DO NOT include '[Not backed by sources]' in the content; the support_status field carries it.\n"
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
    if provider is not None:
        result.metrics.fill_tokens(provider)

    prompt_tokens = (
        getattr(getattr(provider, "last_usage_metadata", None), "prompt_token_count", None)
        or (result.metrics.prompt_tokens if result.metrics else 0)
        or (len(prompt) // 4)
    )
    logger.info(
        "stage5_prompt_token_count",
        prompt_token_count=prompt_tokens,
        prompt_chars=len(prompt),
        sources_count=len(included_resource_ids),
    )

    logger.info(
        "stage5_concept_mapping",
        topic=result.explanation.topic,
        nodes=len(result.concept_nodes),
        sections=len(result.explanation.sections),
    )

    return result


