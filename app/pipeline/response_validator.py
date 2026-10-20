"""
app/pipeline/response_validator.py

Cross-cutting validator executed on every DEVSCOUT learning response.
Guarantees:
1. Schema valid (strict Pydantic v2 domain model).
2. Every explanation claim cites a Resource id that exists in the response.
3. Code gate decision respected (scans fenced code blocks and snippet payloads).
4. Confidence and quality metrics present and within [0.0, 1.0].
5. Retry/repair hook (deterministic, no LLM call).
"""
from __future__ import annotations

import re
from typing import Sequence
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.pipeline import (
    Citation,
    CodeGateLevel,
    EvidenceReviewResult,
    Explanation,
    ExplanationSection,
    LearningResponse,
    Resource,
    ResponseWarning,
)
from app.research.discovery import normalize_url
from app.research.scoring import is_source_relevant_to_claim, normalize_whitespace

FENCED_CODE_PATTERN = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)
NOT_BACKED_BY_SOURCES_MARKER = "[Not backed by sources]"

# Substantive token extraction for deterministic support checks
DEFAULT_STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "over", "after", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "but", "if", "then", "else", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will",
    "just", "should", "now", "it", "its", "this", "that", "these", "those",
    "you", "your", "we", "our", "he", "she", "they", "them", "which", "what",
}

TOKEN_PATTERN = re.compile(r"\b[a-zA-Z0-9_-]{3,}\b")


def tokenize_text(text: str) -> set[str]:
    """Extract lowercase substantive tokens of length >= 3, excluding common stop words."""
    if not text:
        return set()
    words = TOKEN_PATTERN.findall(text.lower())
    return {w for w in words if w not in DEFAULT_STOP_WORDS}


def compute_token_overlap(content: str, excerpts: Sequence[str]) -> float:
    """
    Compute token overlap between section content and cited resource excerpts.
    Returns the fraction of substantive content tokens found in the combined excerpts.
    """
    content_tokens = tokenize_text(content)
    if not content_tokens:
        return 0.0

    combined_excerpts = " ".join(excerpts)
    excerpt_tokens = tokenize_text(combined_excerpts)
    if not excerpt_tokens:
        return 0.0

    overlap = content_tokens & excerpt_tokens
    return len(overlap) / len(content_tokens)


class ValidationReport(BaseModel):
    """Validation report summarizing verification results and detected issues."""

    is_valid: bool
    schema_valid: bool = True
    citations_valid: bool = True
    code_gate_respected: bool = True
    quality_metrics_present: bool = True
    issues: list[str] = Field(default_factory=list)
    repaired: bool = False
    weakly_supported_sections: list[str] = Field(default_factory=list)


def extract_fenced_code_blocks(text: str) -> list[str]:
    """Extract all fenced code block contents from a markdown string."""
    if not text:
        return []
    return [match.strip() for match in FENCED_CODE_PATTERN.findall(text)]


def _build_resource_lookup(
    resources: Sequence[Resource],
) -> tuple[set[str], dict[str, str], dict[str, Resource]]:
    """Build fast lookup structures for verified resource IDs, URLs, and objects."""
    known_ids = {r.resource_id for r in resources}
    url_to_id: dict[str, str] = {}
    id_to_resource: dict[str, Resource] = {r.resource_id: r for r in resources}
    for r in resources:
        if r.url:
            clean_url = r.url.strip()
            url_to_id[clean_url] = r.resource_id
            url_to_id[clean_url.rstrip("/")] = r.resource_id
            try:
                norm = normalize_url(clean_url)
                if norm:
                    url_to_id[norm] = r.resource_id
            except Exception:
                pass
    return known_ids, url_to_id, id_to_resource


def calibrate_confidence_score(
    evidence_or_sections: Any = None,
    explanation_or_resources: Any = None,
    resources: Sequence[Resource] | None = None,
    *,
    evidence_result: EvidenceReviewResult | None = None,
    explanation: Explanation | None = None,
    **kwargs: Any,
) -> float:
    """
    Calibrate confidence score based on verified evidence sufficiency and section support.
    - If irrelevant/insufficient sources -> confidence <= 0.35.
    - Adding a supported section strictly never lowers confidence.
    - Never uses model-provided confidence values.
    Supports flexible signatures:
      calibrate_confidence_score(evidence_result, explanation, resources)
      calibrate_confidence_score(explanation, resources)
      calibrate_confidence_score(sections, resources)
    """
    if evidence_result is not None:
        evidence_or_sections = evidence_result
    if explanation is not None:
        explanation_or_resources = explanation

    evidence_result: EvidenceReviewResult | None = None
    explanation: Explanation | None = None
    res_list: Sequence[Resource] = []

    if isinstance(evidence_or_sections, EvidenceReviewResult):
        evidence_result = evidence_or_sections
        explanation = explanation_or_resources
        res_list = resources or []
    elif isinstance(evidence_or_sections, Explanation):
        explanation = evidence_or_sections
        res_list = explanation_or_resources or []
    elif isinstance(evidence_or_sections, (list, tuple)):
        if evidence_or_sections and isinstance(evidence_or_sections[0], ExplanationSection):
            sections = list(evidence_or_sections)
            res_list = explanation_or_resources or []
            total_sections = len(sections)
            supported_sections = sum(1 for s in sections if s.support_status == "supported")
            support_ratio = (supported_sections / total_sections) if total_sections > 0 else 0.0

            valid_resources = [r for r in res_list if not getattr(r, "is_rejected", False)]
            if not valid_resources:
                score = 0.15 + 0.20 * support_ratio
                return round(min(0.35, score), 2)

            avg_evidence = sum(r.evidence_score for r in valid_resources) / len(valid_resources)
            base = 0.50 + 0.30 * min(1.0, avg_evidence)
            score = base + 0.20 * support_ratio
            return round(max(0.0, min(1.0, score)), 2)
        else:
            res_list = evidence_or_sections
    else:
        explanation = explanation_or_resources
        res_list = resources or []

    total_sections = len(explanation.sections) if explanation and explanation.sections else 0
    supported_sections = (
        sum(1 for s in explanation.sections if s.support_status == "supported")
        if explanation and explanation.sections
        else 0
    )
    support_ratio = (supported_sections / total_sections) if total_sections > 0 else 0.0

    valid_resources = [r for r in (res_list or []) if not getattr(r, "is_rejected", False)]
    is_sufficient = (
        evidence_result.is_sufficient
        if evidence_result is not None
        else (len(valid_resources) >= 2)
    )

    if not is_sufficient or not valid_resources:
        # Confidence <= 0.35 when sources are insufficient
        score = 0.15 + 0.20 * support_ratio
        return round(min(0.35, score), 2)

    # When sources are sufficient
    avg_evidence = sum(r.evidence_score for r in valid_resources) / len(valid_resources)
    base = 0.50 + 0.30 * min(1.0, avg_evidence)
    score = base + 0.20 * support_ratio
    return round(max(0.0, min(1.0, score)), 2)


def validate_explanation(
    explanation: Explanation,
    resources: Sequence[Resource],
    support_threshold: float | None = None,
) -> ValidationReport:
    """
    Validate an Explanation against verified Resources.
    Rules:
    - Every citation carries supporting_quote (max 15 words) occurring verbatim in source text.
    - Missing or not found -> dropped; section with no valid citation -> support_status="unsupported".
    - Citing a rejected resource ID fails.
    - Citing an irrelevant source for a claim fails.
    - Confidence score must be in [0.0, 1.0].
    """
    actual_support_threshold = (
        support_threshold
        if support_threshold is not None
        else get_settings().support_token_overlap_threshold
    )
    issues: list[str] = []
    weakly_supported_sections: list[str] = []
    citations_valid = True
    quality_metrics_present = True

    if not explanation.sections:
        citations_valid = False
        issues.append(
            "Explanation has no sections. Claims must be organized in sections with citations."
        )

    known_ids, url_to_id, id_to_resource = _build_resource_lookup(resources)

    for idx, section in enumerate(explanation.sections):
        valid_cites: list[str] = []
        valid_citations: list[Citation] = []
        phantom_cites: list[str] = []

        # Gather citations to validate
        cits_to_check: list[Citation] = []
        cits_by_id = {c.resource_id: c for c in section.citations}
        if section.citation_resource_ids is not None and len(section.citation_resource_ids) > 0:
            for rid in section.citation_resource_ids:
                if rid in cits_by_id:
                    cits_to_check.append(cits_by_id[rid])
                else:
                    cits_to_check.append(Citation(resource_id=rid, supporting_quote=""))
        elif section.citation_resource_ids is not None and len(section.citation_resource_ids) == 0:
            cits_to_check = []
        elif section.citations:
            cits_to_check = list(section.citations)

        for cit in cits_to_check:
            clean_cite = cit.resource_id.strip()
            matched_id: str | None = None
            if clean_cite in known_ids:
                matched_id = clean_cite
            elif clean_cite in url_to_id:
                matched_id = url_to_id[clean_cite]
            else:
                try:
                    norm = normalize_url(clean_cite)
                    if norm in url_to_id:
                        matched_id = url_to_id[norm]
                except Exception:
                    pass

            if not matched_id or matched_id not in id_to_resource:
                phantom_cites.append(clean_cite)
                continue

            res = id_to_resource[matched_id]

            # 1. Check if resource was rejected
            if getattr(res, "is_rejected", False):
                issues.append(
                    f"Section '{section.title}' (index {idx}) cites rejected resource id '{matched_id}'."
                )
                continue

            # 2. Check relevance to claim
            if not is_source_relevant_to_claim(res, section.title, section.content):
                issues.append(
                    f"Section '{section.title}' (index {idx}) cites source '{res.title}' irrelevant to this claim."
                )
                continue

            # 3. Check quote-anchored citation
            quote = normalize_whitespace(cit.supporting_quote) if cit.supporting_quote else ""
            fetched_text = normalize_whitespace(res.fetched_text)

            if fetched_text or quote:
                if not quote:
                    issues.append(
                        f"Section '{section.title}' (index {idx}) citation to '{matched_id}' missing supporting quote."
                    )
                    continue
                words = quote.split()
                if len(words) > 15:
                    issues.append(
                        f"Section '{section.title}' (index {idx}) supporting quote exceeds 15 words ({len(words)} words)."
                    )
                    continue
                target_text = fetched_text or normalize_whitespace(res.excerpt or res.summary or "")
                if quote.lower() not in target_text.lower():
                    issues.append(
                        f"Section '{section.title}' (index {idx}) supporting quote '{quote}' not found verbatim in resource '{res.title}'."
                    )
                    continue

            # Citation is valid!
            valid_citations.append(Citation(resource_id=matched_id, supporting_quote=quote))
            if matched_id not in valid_cites:
                valid_cites.append(matched_id)

        # Update section with valid citations
        section.citations = valid_citations
        section.citation_resource_ids = valid_cites

        if phantom_cites:
            citations_valid = False
            for pc in phantom_cites:
                issues.append(
                    f"Section '{section.title}' (index {idx}) cites phantom / nonexistent resource id or url '{pc}'."
                )

        has_unbacked_marker = (
            NOT_BACKED_BY_SOURCES_MARKER.lower() in section.content.lower()
            or "not backed by sources" in section.content.lower()
        )

        if not valid_cites:
            section.support_status = "unsupported"
            section.weakly_supported = False
            if not has_unbacked_marker:
                citations_valid = False
                issues.append(
                    f"Section '{section.title}' (index {idx}) has no citations and is not marked 'not backed by sources'."
                )
        else:
            section.support_status = "supported"

        # Deterministic support check: token overlap with cited resources' excerpts
        cited_resources = [id_to_resource[cid] for cid in valid_cites if cid in id_to_resource]
        cited_excerpts = [r.excerpt for r in cited_resources if r.excerpt and r.excerpt.strip()]

        if cited_excerpts and section.support_status == "supported":
            overlap = compute_token_overlap(section.content, cited_excerpts)
            if overlap < actual_support_threshold:
                section.weakly_supported = True
                section.support_status = "weakly_supported"
                weakly_supported_sections.append(section.title)
                issues.append(
                    f"Section '{section.title}' (index {idx}) is weakly supported by cited excerpts "
                    f"(token overlap {overlap:.2f} < {actual_support_threshold:.2f}). [Note: weak signal, not verification]"
                )
            else:
                section.weakly_supported = False
        elif not cited_excerpts:
            section.weakly_supported = False

    if not (0.0 <= explanation.confidence_score <= 1.0):
        quality_metrics_present = False
        issues.append(
            f"Invalid explanation confidence_score: {explanation.confidence_score}. Must be in [0.0, 1.0]."
        )

    is_valid = citations_valid and quality_metrics_present

    return ValidationReport(
        is_valid=is_valid,
        schema_valid=True,
        citations_valid=citations_valid,
        code_gate_respected=True,
        quality_metrics_present=quality_metrics_present,
        issues=issues,
        weakly_supported_sections=weakly_supported_sections,
    )



def repair_explanation(
    explanation: Explanation,
    resources: Sequence[Resource],
    report: ValidationReport | None = None,
) -> tuple[Explanation, ValidationReport]:
    """
    Deterministic repair hook for explanations:
    - Translates verified Resource URLs in citation_resource_ids to their resource_id.
    - Strips phantom citations not in verified resources.
    - If a section has no valid citations, marks its content with '[Not backed by sources]'.
    - Clamps confidence_score to [0.0, 1.0].
    """
    known_ids, url_to_id, _ = _build_resource_lookup(resources)

    has_url_citations = False
    for sec in explanation.sections:
        for cite in sec.citation_resource_ids:
            clean = cite.strip()
            if clean in url_to_id and clean not in known_ids:
                has_url_citations = True
                break
        if has_url_citations:
            break

    if report is not None and report.is_valid and not has_url_citations:
        return explanation, report

    exp_data = explanation.model_dump()
    sections_data = exp_data.get("sections", [])

    for sec in sections_data:
        raw_cites = sec.get("citation_resource_ids", [])
        valid_cites: list[str] = []
        for cite in raw_cites:
            clean_cite = cite.strip()
            if clean_cite in known_ids:
                if clean_cite not in valid_cites:
                    valid_cites.append(clean_cite)
            elif clean_cite in url_to_id:
                target_id = url_to_id[clean_cite]
                if target_id not in valid_cites:
                    valid_cites.append(target_id)
            else:
                try:
                    norm = normalize_url(clean_cite)
                    if norm in url_to_id:
                        target_id = url_to_id[norm]
                        if target_id not in valid_cites:
                            valid_cites.append(target_id)
                except Exception:
                    pass

        sec["citation_resource_ids"] = valid_cites

        # If no valid citations remain, mark as [Not backed by sources]
        if not valid_cites:
            content = sec.get("content", "")
            if (
                NOT_BACKED_BY_SOURCES_MARKER.lower() not in content.lower()
                and "not backed by sources" not in content.lower()
            ):
                sec["content"] = f"{content.strip()} {NOT_BACKED_BY_SOURCES_MARKER}".strip()

    def _clamp(val: float | None) -> float:
        if val is None:
            return 1.0
        return max(0.0, min(1.0, float(val)))

    exp_data["confidence_score"] = _clamp(exp_data.get("confidence_score"))

    repaired_exp = Explanation.model_validate(exp_data)
    new_report = validate_explanation(repaired_exp, resources)
    new_report.repaired = True
    return repaired_exp, new_report


def validate_response(response: LearningResponse) -> ValidationReport:
    """
    Validate a LearningResponse against all integrity and policy rules.
    """
    issues: list[str] = []
    schema_valid = True
    citations_valid = True
    code_gate_respected = True
    quality_metrics_present = True

    # 1. Schema check
    if not isinstance(response, LearningResponse):
        schema_valid = False
        issues.append("Response is not an instance of LearningResponse.")
        return ValidationReport(
            is_valid=False,
            schema_valid=False,
            citations_valid=False,
            code_gate_respected=False,
            quality_metrics_present=False,
            issues=issues,
        )

    weakly_supported_sections: list[str] = []

    # 2. Citations check: every explanation claim must cite an existing Resource id/url or be marked not backed by sources
    if response.explanation:
        exp_report = validate_explanation(response.explanation, response.resources)
        if not exp_report.citations_valid:
            citations_valid = False
        issues.extend(exp_report.issues)
        weakly_supported_sections.extend(exp_report.weakly_supported_sections)

    # 3. Code gate decision respected

    gate_level = response.code_gate_decision.level

    # Collect all fenced code blocks in text fields
    all_text_blocks: list[str] = []
    if response.explanation:
        all_text_blocks.extend(extract_fenced_code_blocks(response.explanation.summary))
        for sec in response.explanation.sections:
            all_text_blocks.extend(extract_fenced_code_blocks(sec.content))

    all_snippets = response.snippets
    total_code_items = len(all_text_blocks) + len(all_snippets)

    if gate_level in (CodeGateLevel.EXPLAIN, CodeGateLevel.REQUIRE_CHECK):
        if total_code_items > 0:
            code_gate_respected = False
            issues.append(
                f"Gate level is {gate_level}, but found {len(all_text_blocks)} fenced code block(s) "
                f"and {len(all_snippets)} snippet(s). No code may be returned under {gate_level}."
            )

    elif gate_level == CodeGateLevel.SNIPPET:
        if total_code_items > 3:
            code_gate_respected = False
            issues.append(
                f"Gate level is SNIPPET, but found {total_code_items} code block(s)/snippet(s). Max allowed is 3."
            )
        for idx, block in enumerate(all_text_blocks):
            lines = len(block.splitlines()) if block else 0
            if lines > 15:
                code_gate_respected = False
                issues.append(
                    f"Fenced code block {idx + 1} in text exceeds 15 lines (got {lines} lines)."
                )
        for idx, snip in enumerate(all_snippets):
            lines = len(snip.code.strip().splitlines())
            if lines > 15:
                code_gate_respected = False
                issues.append(
                    f"Snippet model {idx + 1} exceeds 15 lines (got {lines} lines)."
                )

    # 4. Confidence & quality scores present and valid
    if response.confidence_score is None or not (0.0 <= response.confidence_score <= 1.0):
        quality_metrics_present = False
        issues.append(f"Invalid response confidence_score: {response.confidence_score}. Must be in [0.0, 1.0].")

    if response.quality_score is None or not (0.0 <= response.quality_score <= 1.0):
        quality_metrics_present = False
        issues.append(f"Invalid response quality_score: {response.quality_score}. Must be in [0.0, 1.0].")

    if response.explanation and not (0.0 <= response.explanation.confidence_score <= 1.0):
        quality_metrics_present = False
        issues.append(
            f"Invalid explanation confidence_score: {response.explanation.confidence_score}. Must be in [0.0, 1.0]."
        )

    # 5. Calibrate confidence score (scores never come from model-provided values)
    if response.explanation:
        calibrated = calibrate_confidence_score(
            evidence_or_sections=None,
            explanation_or_resources=response.explanation,
            resources=response.resources,
        )
        response.explanation.confidence_score = calibrated
        response.confidence_score = calibrated

    # 6. Populate warnings: UNSUPPORTED_CLAIMS and NO_RELIABLE_SOURCES
    if response.explanation:
        if any(sec.support_status == "unsupported" for sec in response.explanation.sections):
            if not any(w.code == "UNSUPPORTED_CLAIMS" for w in response.warnings):
                response.warnings.append(
                    ResponseWarning(
                        severity="warning",
                        code="UNSUPPORTED_CLAIMS",
                        message="One or more explanation sections contain unsupported claims.",
                    )
                )

    valid_resources = [r for r in response.resources if not getattr(r, "is_rejected", False)]
    if not valid_resources:
        if not any(w.code == "NO_RELIABLE_SOURCES" for w in response.warnings):
            response.warnings.append(
                ResponseWarning(
                    severity="critical",
                    code="NO_RELIABLE_SOURCES",
                    message="Evidence is insufficient to support reliable answers.",
                )
            )

    is_valid = schema_valid and citations_valid and code_gate_respected and quality_metrics_present

    return ValidationReport(
        is_valid=is_valid,
        schema_valid=schema_valid,
        citations_valid=citations_valid,
        code_gate_respected=code_gate_respected,
        quality_metrics_present=quality_metrics_present,
        issues=issues,
        weakly_supported_sections=weakly_supported_sections,
    )



def repair_response(response: LearningResponse, report: ValidationReport) -> tuple[LearningResponse, ValidationReport]:
    """
    Deterministic repair hook for invalid responses.
    - Strips unauthorized fenced code blocks when gate is EXPLAIN or REQUIRE_CHECK.
    - Resolves URLs or filters out phantom citations, marking unbacked claims as not backed by sources.
    - Clamps confidence and quality scores.
    """
    if report.is_valid:
        return response, report

    # Copy / clone response for repair
    repaired_data = response.model_dump()
    repaired_data["resources"] = [r.model_copy() for r in response.resources]
    gate_level = response.code_gate_decision.level

    # 1. Repair code gate violations
    if gate_level in (CodeGateLevel.EXPLAIN, CodeGateLevel.REQUIRE_CHECK):
        # Clear snippets
        repaired_data["snippets"] = []
        # Strip fenced code blocks from explanation
        if repaired_data.get("explanation"):
            exp = repaired_data["explanation"]
            exp["summary"] = FENCED_CODE_PATTERN.sub("[Code omitted: conceptual explanation only]", exp.get("summary", ""))
            for sec in exp.get("sections", []):
                sec["content"] = FENCED_CODE_PATTERN.sub("[Code omitted: conceptual explanation only]", sec.get("content", ""))

    elif gate_level == CodeGateLevel.SNIPPET:
        # Keep at most 3 snippets, truncate lines if needed
        valid_snippets = []
        for snip in repaired_data.get("snippets", [])[:3]:
            lines = snip["code"].strip().splitlines()
            if len(lines) > 15:
                snip["code"] = "\n".join(lines[:15])
                snip["line_count"] = 15
            valid_snippets.append(snip)
        repaired_data["snippets"] = valid_snippets

    # 2. Repair citation violations
    if repaired_data.get("explanation"):
        current_exp = Explanation.model_validate(repaired_data["explanation"])
        repaired_exp, _ = repair_explanation(current_exp, response.resources)
        repaired_data["explanation"] = repaired_exp.model_dump()

    # 3. Repair metrics & scores
    def _clamp(val: float | None) -> float:
        if val is None:
            return 1.0
        return max(0.0, min(1.0, float(val)))

    repaired_data["confidence_score"] = _clamp(repaired_data.get("confidence_score"))
    repaired_data["quality_score"] = _clamp(repaired_data.get("quality_score"))
    if repaired_data.get("explanation"):
        repaired_data["explanation"]["confidence_score"] = _clamp(
            repaired_data["explanation"].get("confidence_score")
        )

    repaired_response = LearningResponse.model_validate(repaired_data)
    new_report = validate_response(repaired_response)
    new_report.repaired = True
    return repaired_response, new_report

