"""
Tests for app/policy/code_gate.py and app/policy/attempt_review.py.

Covers:
- explain default
- 15-line snippet ok
- 16-line snippet rejected
- >3 snippets rejected
- full code without readiness -> REQUIRE_CHECK
- trivial attempt does NOT unlock FULL
- copied snippet does NOT unlock FULL
- invalid syntax attempt does NOT unlock FULL
- acceptable attempt unlocks FULL
- each readiness path (understanding check, acceptable attempt, final build)
- no explicit request -> EXPLAIN
- snippet missing explanation rejected
- configurable readiness rule
- attempt_review logic (min lines, blanks/comments ignored, SHA-256 hash match, AST check)
"""
import pytest

from app.models.pipeline import (
    AttemptReview,
    AttemptStatus,
    CodeGateLevel,
    Snippet,
)
from app.policy.attempt_review import (
    count_real_lines,
    hash_code,
    normalize_code,
    review_attempt,
)
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
    with pytest.raises(Exception):
        # Snippet Pydantic validation rejects >15 lines
        Snippet(
            code=code_16_lines,
            concept="Too long",
            line_count=16,
            explanation="Explanation text",
        )

    # In evaluate_code_gate directly:
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
            attempt_review=None,
            final_build_submitted=False,
        ),
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK
    assert any("readiness criteria not met" in r for r in decision.reasons)


def test_trivial_attempt_does_not_unlock_full():
    """Trivial attempt (e.g. print('x')) does not satisfy readiness and returns REQUIRE_CHECK."""
    trivial_code = "print('x')"
    review = review_attempt(trivial_code)
    assert review.status == AttemptStatus.TRIVIAL

    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(attempt_review=review),
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK
    assert any("readiness criteria not met" in r for r in decision.reasons)


def test_copied_snippet_does_not_unlock_full():
    """Attempt that copies a previously provided snippet triggers COPIED_SNIPPET and does not unlock FULL."""
    snippet_code = "x = 1\ny = 2\nresult = x + y"
    snippet_hash = hash_code(snippet_code)

    learner_attempt = "  x = 1\n  y = 2\n  result = x + y\n"
    review = review_attempt(
        learner_attempt,
        previous_snippet_hashes=[snippet_hash],
    )
    assert review.status == AttemptStatus.COPIED_SNIPPET

    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(attempt_review=review),
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK


def test_invalid_syntax_does_not_unlock_full():
    """Attempt with invalid Python syntax triggers INVALID and does not unlock FULL."""
    broken_code = """
def broken_syntax(
    x = 10
    if x > 5
        return True
"""
    review = review_attempt(broken_code)
    assert review.status == AttemptStatus.INVALID

    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(attempt_review=review),
    )
    assert decision.level == CodeGateLevel.REQUIRE_CHECK


def test_acceptable_attempt_unlocks_full():
    """Acceptable substantive attempt with >=10 real lines and valid syntax unlocks FULL."""
    substantive_code = """
# Rate limiter implementation attempt
class TokenBucket:
    def __init__(self, capacity: int, fill_rate: float):
        self.capacity = capacity
        self.fill_rate = fill_rate
        self.tokens = capacity
        self.last_fill = 0.0

    def consume(self, amount: int = 1) -> bool:
        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False
"""
    review = review_attempt(substantive_code)
    assert review.status == AttemptStatus.ACCEPTABLE

    decision = evaluate_code_gate(
        explicit_request=True,
        wants_full_code=True,
        proposed_full_code="def solution(): return 42",
        has_explanation=True,
        readiness=LearnerReadiness(attempt_review=review),
    )
    assert decision.level == CodeGateLevel.FULL
    assert any("readiness verified" in r for r in decision.reasons)


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


def test_attempt_review_comments_and_blanks_ignored():
    """Blanks and comments must not count toward the min real lines requirement."""
    comment_padded_code = """
# Comment line 1
# Comment line 2

# Comment line 3

# Comment line 4
print("only one real line")
"""
    assert count_real_lines(comment_padded_code) == 1
    review = review_attempt(comment_padded_code, min_real_lines=10)
    assert review.status == AttemptStatus.TRIVIAL


def test_ten_lines_of_literal_assignments_is_trivial():
    """10 lines of literal assignments (xN = N) without structural substance must return TRIVIAL."""
    code = "\n".join([f"x{i} = {i}" for i in range(10)])
    review = review_attempt(code, min_real_lines=10)
    assert review.status == AttemptStatus.TRIVIAL
    assert any("structural substance" in r for r in review.reasons)


def test_real_function_with_logic_is_acceptable():
    """A real function with logic (FunctionDef, loop, conditional) >= 10 lines returns ACCEPTABLE."""
    code = """
def process_scores(scores: list[int]) -> dict[str, int]:
    total = 0
    passing_count = 0
    for score in scores:
        if score >= 50:
            passing_count += 1
            total += score
        else:
            total += 0
    average = total // max(len(scores), 1)
    return {"total": total, "passing": passing_count, "avg": average}
"""
    review = review_attempt(code, min_real_lines=10)
    assert review.status == AttemptStatus.ACCEPTABLE


def test_renamed_variable_copy_detected():
    """Copy detection catches renamed-variable AST clones above threshold."""
    snippet = """
def add_item_to_cart(cart, item, quantity):
    if item not in cart:
        cart[item] = quantity
    else:
        cart[item] += quantity
    return cart
"""
    renamed_attempt = """
def store_element_in_basket(basket, element, amount):
    if element not in basket:
        basket[element] = amount
    else:
        basket[element] += amount
    return basket
"""
    review = review_attempt(renamed_attempt, previous_snippets=[snippet])
    assert review.status == AttemptStatus.COPIED_SNIPPET
    assert any("too similar" in r.lower() or "snippet" in r.lower() for r in review.reasons)


def test_junk_code_rejected_as_irrelevant():
    """Syntactically valid structural code that lacks required milestone concepts is rejected as IRRELEVANT."""
    fibonacci_code = """
def calculate_fibonacci(count: int) -> list[int]:
    sequence = [0, 1]
    total_iterations = 0
    for idx in range(2, count):
        next_val = sequence[idx - 1] + sequence[idx - 2]
        if next_val > 1000:
            sequence.append(1000)
        else:
            sequence.append(next_val)
        total_iterations += 1
    return sequence
"""
    # Requires concepts related to caching, but code is fibonacci
    review = review_attempt(
        fibonacci_code,
        milestone_concepts=["token_bucket", "refill_rate", "capacity"],
        min_concept_references=2,
    )
    assert review.status == AttemptStatus.IRRELEVANT
    assert any("relevance" in r.lower() for r in review.reasons)


def test_relevant_code_accepted_with_milestone_concepts():
    """Attempt that references at least K milestone concepts is marked ACCEPTABLE."""
    relevant_code = """
class TokenBucketLimiter:
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = capacity

    def consume(self, count: int = 1) -> bool:
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False
"""
    review = review_attempt(
        relevant_code,
        milestone_concepts=["token_bucket", "refill_rate", "capacity"],
        min_concept_references=2,
    )
    assert review.status == AttemptStatus.ACCEPTABLE

