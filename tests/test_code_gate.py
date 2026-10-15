"""
Tests for app/policy/code_gate.py – Deterministic LLM-free Code Assistance Gate.

Covers:
- explain default
- 15-line snippet ok
- 16-line snippet rejected
- >3 snippets rejected
- full code without readiness -> REQUIRE_CHECK
- full code with each readiness path (understanding check, learner attempt, final build)
- no explicit request -> EXPLAIN
- snippet missing explanation rejected
- full runnable entrypoint in snippet rejected
- configurable readiness rule
"""
import pytest

from app.models.pipeline import CodeGateLevel, Snippet
from app.policy.code_gate import (
    LearnerReadiness,
    ReadinessPolicyConfig,
    evaluate_code_gate,
)


def test_code_gate_default_is_explain():
    """Default decision must be EXPLAIN with no code."""
    decision = evaluate_code_gate()
    assert decision.level == CodeGateLevel.EXPLAIN
    assert any("Default policy is EXPLAIN" in r for r in decision.reasons)
    assert len(decision.violations) == 0


def test_15_line_snippet_ok():
    """Snippets up to 15 lines are allowed under SNIPPET gate when accompanied by explanation."""
    code_15_lines = "\n".join([f"x_{i} = {i}" for i in range(15)])
    snippet = Snippet(
        code=code_15_lines,
        language="python",
        concept="Variable assignment",
        line_count=15,
        explanation="Demonstrates 15 scalar variable assignments.",
    )
    decision = evaluate_code_gate(
        wants_snippet=True,
        proposed_snippets=[snippet],
        has_explanation=True,
    )
    assert decision.level == CodeGateLevel.SNIPPET
    assert len(decision.violations) == 0


def test_16_line_snippet_rejected():
    """Snippet with 16 lines violates policy and drops to EXPLAIN with recorded violation."""
    code_16_lines = "\n".join([f"x_{i} = {i}" for i in range(16)])
    # Pydantic model itself rejects line_count > 15, but let's test evaluation of candidate code
    # We can create a mock or pass a snippet with 16 physical lines:
    with pytest.raises(Exception):
        # Snippet Pydantic validation rejects >15 lines
        Snippet(
            code=code_16_lines,
            concept="Too long",
            line_count=16,
            explanation="Explanation text",
        )

    # In evaluate_code_gate directly:
    # If a snippet object with code > 15 lines was submitted
    # We test evaluate_code_gate's line count inspector by constructing a snippet or passing it:
    snippet_15 = Snippet(
        code="\n".join([f"x_{i} = {i}" for i in range(15)]),
        concept="Valid count",
        line_count=15,
        explanation="Explanation text",
    )
    # Manually assign 16 physical lines to test gate's detection
    snippet_15.code = code_16_lines

    decision = evaluate_code_gate(
        wants_snippet=True,
        proposed_snippets=[snippet_15],
    )
    assert decision.level == CodeGateLevel.EXPLAIN
    assert any("exceeds 15-line limit" in v for v in decision.violations)


def test_more_than_3_snippets_rejected():
    """> 3 snippets exceeds maximum allowed limit and generates a violation."""
    snippets = [
        Snippet(
            code=f"val_{i} = {i}",
            concept=f"Concept {i}",
            line_count=1,
            explanation=f"Explanation for concept {i}",
        )
        for i in range(4)
    ]
    decision = evaluate_code_gate(
        wants_snippet=True,
        proposed_snippets=snippets,
    )
    assert decision.level == CodeGateLevel.EXPLAIN
    assert any("Exceeded maximum allowed snippets" in v for v in decision.violations)


def test_full_code_without_readiness_requires_check():
    """Full code with explicit request but no readiness must return REQUIRE_CHECK."""
    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def build(): pass",
        readiness=LearnerReadiness(
            understanding_check_passed=False,
            learner_attempt_submitted=False,
            final_build_submitted=False,
        ),
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK
    assert any("readiness criteria not met" in r for r in decision.reasons)


def test_full_code_readiness_path_understanding_check():
    """Readiness path 1: Understanding check passed allows FULL code."""
    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(understanding_check_passed=True),
    )
    assert decision.level == CodeGateLevel.FULL
    assert any("readiness verified" in r for r in decision.reasons)


def test_full_code_readiness_path_learner_attempt():
    """Readiness path 2: Learner submitted own attempt allows FULL code."""
    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(learner_attempt_submitted=True),
    )
    assert decision.level == CodeGateLevel.FULL
    assert any("readiness verified" in r for r in decision.reasons)


def test_full_code_readiness_path_final_build():
    """Readiness path 3: Final build submitted allows FULL code."""
    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(final_build_submitted=True),
    )
    assert decision.level == CodeGateLevel.FULL
    assert any("readiness verified" in r for r in decision.reasons)


def test_full_code_no_explicit_request_defaults_to_explain():
    """If full code is contemplated but learner did not explicitly request it, gate returns EXPLAIN."""
    decision = evaluate_code_gate(
        explicit_request=False,
        wants_full_code=True,
        readiness=LearnerReadiness(understanding_check_passed=True),
    )
    assert decision.level == CodeGateLevel.EXPLAIN
    assert any("not explicitly requested" in r for r in decision.reasons)


def test_snippet_missing_explanation_rejected():
    """Returned code without explanation triggers a violation."""
    snip = Snippet(
        code="x = 10",
        concept="Scalar",
        line_count=1,
        explanation="Valid length explanation here",
    )
    decision = evaluate_code_gate(
        wants_snippet=True,
        proposed_snippets=[snip],
        has_explanation=False,
    )
    assert decision.level == CodeGateLevel.EXPLAIN
    assert any("pedagogical explanation" in v for v in decision.violations)


def test_configurable_readiness_rule():
    """Readiness rule can be custom configured in one place."""
    # Custom rule: only allow final build, disallow understanding check
    strict_config = ReadinessPolicyConfig(
        allow_understanding_check=False,
        allow_learner_attempt=False,
        allow_final_build=True,
    )
    readiness = LearnerReadiness(understanding_check_passed=True, final_build_submitted=False)
    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        readiness=readiness,
        config=strict_config,
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK

    # Now provide final build under the same strict config
    readiness_build = LearnerReadiness(final_build_submitted=True)
    decision_ok = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        has_explanation=True,
        readiness=readiness_build,
        config=strict_config,
    )
    assert decision_ok.level == CodeGateLevel.FULL
