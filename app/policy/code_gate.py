"""
app/policy/code_gate.py

Deterministic, LLM-free Code Assistance Gate for DEVSCOUT.
Enforces the core principle: "Learning over copy-paste."

Rules:
- Default: EXPLAIN. No code.
- SNIPPET allowed when useful: max 15 lines, one concept, never a complete runnable file or full wiring.
- FULL implementation only if explicit_request is true AND readiness is met.
  Readiness = understanding check passed OR learner submitted their own attempt OR final build submitted.
  Readiness rule is configurable in one place via ReadinessPolicyConfig.
- If full code is requested but readiness is not met: return REQUIRE_CHECK, not code.
- All returned code must come with an explanation.
"""
from __future__ import annotations

import re
from pydantic import BaseModel, Field

from app.models.pipeline import CodeGateDecision, CodeGateLevel, Snippet

MAX_SNIPPET_LINES = 15
MAX_SNIPPET_COUNT = 3

FULL_WIRING_PATTERNS = [
    re.compile(r"if\s+__name__\s*==\s*['\"]__main__['\"]"),
    re.compile(r"app\s*=\s*(?:FastAPI|Flask|Express)\b"),
    re.compile(r"def\s+main\s*\(\)\s*->\s*None:"),
    re.compile(r"uvicorn\.run\("),
]


class LearnerReadiness(BaseModel):
    """Signals reflecting learner progress towards code readiness."""

    understanding_check_passed: bool = False
    learner_attempt_submitted: bool = False
    final_build_submitted: bool = False


class ReadinessPolicyConfig(BaseModel):
    """
    Configurable readiness policy in one central place.
    Readiness = understanding check passed OR learner submitted attempt OR final build submitted.
    """

    allow_understanding_check: bool = True
    allow_learner_attempt: bool = True
    allow_final_build: bool = True

    def is_ready(self, readiness: LearnerReadiness | None) -> bool:
        """Evaluate whether learner meets readiness criteria for full code."""
        if readiness is None:
            return False
        if self.allow_understanding_check and readiness.understanding_check_passed:
            return True
        if self.allow_learner_attempt and readiness.learner_attempt_submitted:
            return True
        if self.allow_final_build and readiness.final_build_submitted:
            return True
        return False


# Singleton default readiness policy configuration
DEFAULT_READINESS_CONFIG = ReadinessPolicyConfig()


def is_full_runnable_or_wiring(code: str) -> bool:
    """Detect if code snippet contains full runnable application entrypoints or complete wiring."""
    for pattern in FULL_WIRING_PATTERNS:
        if pattern.search(code):
            return True
    return False


def count_code_lines(code: str) -> int:
    """Count non-empty/stripped lines in code block."""
    stripped = code.strip()
    if not stripped:
        return 0
    return len(stripped.splitlines())


def evaluate_code_gate(
    explicit_request: bool = False,
    wants_full_code: bool = False,
    wants_snippet: bool = False,
    readiness: LearnerReadiness | None = None,
    proposed_snippets: list[Snippet] | None = None,
    proposed_full_code: str | None = None,
    has_explanation: bool = True,
    config: ReadinessPolicyConfig | None = None,
) -> CodeGateDecision:
    """
    Evaluate candidate code release against DEVSCOUT Code Policy.

    Returns:
        CodeGateDecision with level EXPLAIN | SNIPPET | FULL | REQUIRE_CHECK,
        reasons explaining the decision, and any policy violations detected.
    """
    policy_config = config or DEFAULT_READINESS_CONFIG
    reasons: list[str] = []
    violations: list[str] = []
    snippets = proposed_snippets or []

    # 1. Inspect candidate snippets for policy violations
    if len(snippets) > MAX_SNIPPET_COUNT:
        violations.append(
            f"Exceeded maximum allowed snippets: received {len(snippets)}, max is {MAX_SNIPPET_COUNT}."
        )

    for idx, snip in enumerate(snippets):
        lines = count_code_lines(snip.code)
        if lines > MAX_SNIPPET_LINES:
            violations.append(
                f"Snippet {idx + 1} exceeds {MAX_SNIPPET_LINES}-line limit (got {lines} lines)."
            )
        if is_full_runnable_or_wiring(snip.code):
            violations.append(
                f"Snippet {idx + 1} contains full runnable entrypoint or wiring, which is disallowed for snippets."
            )
        if not snip.explanation or len(snip.explanation.strip()) < 5:
            violations.append(f"Snippet {idx + 1} is missing a pedagogical explanation.")

    # Check overall explanation requirement for any returned code
    if (snippets or (wants_full_code and proposed_full_code)) and not has_explanation:
        violations.append("All returned code must come with a pedagogical explanation.")

    # 2. Case: Full Implementation Requested
    if wants_full_code:
        if not explicit_request:
            reasons.append("Full code was not explicitly requested by learner. Defaulting to EXPLAIN.")
            return CodeGateDecision(
                level=CodeGateLevel.EXPLAIN,
                reasons=reasons,
                violations=violations,
            )

        is_ready = policy_config.is_ready(readiness)
        if not is_ready:
            reasons.append(
                "Full implementation requested but learner readiness criteria not met. "
                "Must pass understanding check, submit own attempt, or submit final build."
            )
            return CodeGateDecision(
                level=CodeGateLevel.REQUIRE_CHECK,
                reasons=reasons,
                violations=violations,
            )

        # Explicit request is True AND learner is ready
        if violations:
            reasons.append("Learner is ready for full code, but candidate code violated policy constraints.")
            return CodeGateDecision(
                level=CodeGateLevel.EXPLAIN,
                reasons=reasons,
                violations=violations,
            )

        reasons.append(
            "Explicit request received and learner readiness verified. Full implementation permitted."
        )
        return CodeGateDecision(
            level=CodeGateLevel.FULL,
            reasons=reasons,
            violations=violations,
        )

    # 3. Case: Snippet Requested / Permitted
    if wants_snippet:
        if violations:
            reasons.append("Snippet policy violated; falling back to EXPLAIN.")
            return CodeGateDecision(
                level=CodeGateLevel.EXPLAIN,
                reasons=reasons,
                violations=violations,
            )

        reasons.append(
            "Targeted single-concept snippet (<= 15 lines) permitted with pedagogical explanation."
        )
        return CodeGateDecision(
            level=CodeGateLevel.SNIPPET,
            reasons=reasons,
            violations=violations,
        )

    # 4. Default Case: EXPLAIN. No code.
    reasons.append("Default policy is EXPLAIN: guide conceptual understanding without returning raw code.")
    return CodeGateDecision(
        level=CodeGateLevel.EXPLAIN,
        reasons=reasons,
        violations=violations,
    )
