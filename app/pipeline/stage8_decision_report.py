"""
Pipeline Stage 8 – Code Assistance Gate.

Responsibility: Enforce the pedagogical code gate policy. Decides whether candidate
code (or snippet) may be released to the learner or if an understanding check is required.
"""
from __future__ import annotations

from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    CodeGateDecision,
    CodeGateLevel,
    CodeGateResult,
    Snippet,
    StageMetrics,
)
from app.policy.code_gate import (
    LearnerReadiness,
    ReadinessPolicyConfig,
    evaluate_code_gate,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 8
STAGE_NAME = "Code Assistance Gate"


async def run(
    explicit_request: bool = False,
    wants_full_code: bool = False,
    wants_snippet: bool = False,
    readiness: LearnerReadiness | None = None,
    proposed_snippets: list[Snippet] | None = None,
    proposed_full_code: str | None = None,
    has_explanation: bool = True,
    config: ReadinessPolicyConfig | None = None,
) -> CodeGateResult:
    """
    Stage 8 entry point: Evaluate code gate policy and determine permitted code level.
    """
    started_at = datetime.now(UTC)

    decision: CodeGateDecision = evaluate_code_gate(
        explicit_request=explicit_request,
        wants_full_code=wants_full_code,
        wants_snippet=wants_snippet,
        readiness=readiness,
        proposed_snippets=proposed_snippets,
        proposed_full_code=proposed_full_code,
        has_explanation=has_explanation,
        config=config,
    )

    allowed_snippets: list[Snippet] = []
    if decision.level == CodeGateLevel.SNIPPET and proposed_snippets:
        allowed_snippets = proposed_snippets[:3]

    allowed_full_code: str | None = None
    if decision.level == CodeGateLevel.FULL:
        allowed_full_code = proposed_full_code

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage8_code_gate",
        level=decision.level,
        violations_count=len(decision.violations),
    )

    return CodeGateResult(
        decision=decision,
        allowed_snippets=allowed_snippets,
        full_code=allowed_full_code,
        metrics=metrics,
    )
