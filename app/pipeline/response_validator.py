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
from pydantic import BaseModel, Field

from app.models.pipeline import (
    CodeGateLevel,
    LearningResponse,
    Resource,
)

FENCED_CODE_PATTERN = re.compile(r"```(?:\w+)?\s*\n?(.*?)```", re.DOTALL)


class ValidationReport(BaseModel):
    """Validation report summarizing verification results and detected issues."""

    is_valid: bool
    schema_valid: bool = True
    citations_valid: bool = True
    code_gate_respected: bool = True
    quality_metrics_present: bool = True
    issues: list[str] = Field(default_factory=list)
    repaired: bool = False


def extract_fenced_code_blocks(text: str) -> list[str]:
    """Extract all fenced code block contents from a markdown string."""
    if not text:
        return []
    return [match.strip() for match in FENCED_CODE_PATTERN.findall(text)]


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

    # 2. Citations check: every explanation claim must cite an existing Resource id
    known_resource_ids = {r.resource_id for r in response.resources}
    if response.explanation:
        if not response.explanation.sections:
            citations_valid = False
            issues.append("Explanation has no sections. Claims must be organized in sections with citations.")
        for idx, section in enumerate(response.explanation.sections):
            if not section.citation_resource_ids:
                citations_valid = False
                issues.append(
                    f"Section '{section.title}' (index {idx}) has no citations. "
                    "Every claim must cite at least one verified Resource id."
                )
            else:
                for cite_id in section.citation_resource_ids:
                    if cite_id not in known_resource_ids:
                        citations_valid = False
                        issues.append(
                            f"Section '{section.title}' cites nonexistent resource id '{cite_id}'."
                        )

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

    is_valid = schema_valid and citations_valid and code_gate_respected and quality_metrics_present

    return ValidationReport(
        is_valid=is_valid,
        schema_valid=schema_valid,
        citations_valid=citations_valid,
        code_gate_respected=code_gate_respected,
        quality_metrics_present=quality_metrics_present,
        issues=issues,
    )


def repair_response(response: LearningResponse, report: ValidationReport) -> tuple[LearningResponse, ValidationReport]:
    """
    Deterministic repair hook for invalid responses.
    - Strips unauthorized fenced code blocks when gate is EXPLAIN or REQUIRE_CHECK.
    - Filters out nonexistent resource citation IDs.
    - Clamps confidence and quality scores.
    """
    if report.is_valid:
        return response, report

    # Copy / clone response for repair
    repaired_data = response.model_dump()
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
    known_resource_ids = {r.resource_id for r in response.resources}
    if repaired_data.get("explanation"):
        exp = repaired_data["explanation"]
        for sec in exp.get("sections", []):
            sec["citation_resource_ids"] = [
                cid for cid in sec.get("citation_resource_ids", [])
                if cid in known_resource_ids
            ]
            # If section citations became empty but resources exist, link the first resource
            if not sec["citation_resource_ids"] and known_resource_ids:
                sec["citation_resource_ids"] = [next(iter(known_resource_ids))]

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
