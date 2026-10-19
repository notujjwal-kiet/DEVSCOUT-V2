"""
Pipeline Stage 4 – Evidence Review & Validation.

Responsibility: Deterministically review, score, and validate discovered sources:
- Scores each resource using score_document from app.research.scoring (freshness, reliability).
- Performs sufficiency checks against configurable thresholds:
  1. Minimum distinct domains (diversity of sources)
  2. Minimum distinct source types (multi-perspective diversity)
  3. At least one authoritative official docs or repo source
- Flags uncertainty (uncertainty_flag=True) for any resource when:
  1. Its date is unknown (last_updated is None)
  2. It is stale for its resource type (age > half_life or freshness < threshold)
  3. It is the sole source for a topic (under-evidenced singleton topic)
- Evaluates loop trigger: if insufficient and loop_count < MAX_EVIDENCE_RETRY_LOOPS,
  triggers retry loop and returns seen_urls for exclusion in Stage 3.
- Deterministic only: no LLM calls, no invented numeric confidence.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Sequence

import structlog

from app.models.pipeline import (
    EvidenceReviewResult,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
    StageMetrics,
)
from app.research.models import ResearchDocument
from app.research.scoring import (
    extract_domain,
    get_half_life_days,
    is_official_docs_domain,
    score_document,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 4
STAGE_NAME = "Evidence Review"

# ── Config Block (Thresholds & Constants) ────────────────────────────────────
DEFAULT_MIN_DISTINCT_DOMAINS: int = 2
DEFAULT_MIN_DISTINCT_SOURCE_TYPES: int = 2
REQUIRE_OFFICIAL_DOCS_OR_REPO: bool = True
DEFAULT_FRESHNESS_STALE_THRESHOLD: float = 0.5
MAX_EVIDENCE_RETRY_LOOPS: int = 2


async def run(
    discovery_result: SourceDiscoveryResult,
    freshness_threshold: float = DEFAULT_FRESHNESS_STALE_THRESHOLD,
    min_distinct_domains: int = DEFAULT_MIN_DISTINCT_DOMAINS,
    min_distinct_source_types: int = DEFAULT_MIN_DISTINCT_SOURCE_TYPES,
    require_official_or_repo: bool = REQUIRE_OFFICIAL_DOCS_OR_REPO,
    max_loops: int = MAX_EVIDENCE_RETRY_LOOPS,
    loop_count: int | None = None,
    max_retry_loops: int | None = None,
    now: datetime | None = None,
) -> EvidenceReviewResult:
    """
    Stage 4 entry point: Review and score evidence; trigger loop 4->3 if insufficient.
    Deterministic only (no LLM).
    """
    started_at = datetime.now(UTC)
    current_time = now if now is not None else started_at
    goal_id = discovery_result.goal_id
    if max_retry_loops is not None:
        max_loops = max_retry_loops
    active_loop_count = loop_count if loop_count is not None else discovery_result.loop_count
    loop_count = active_loop_count


    flagged_uncertainties: list[str] = []
    reviewed_resources: list[Resource] = []

    # 1. Deterministic scoring of each resource using score_document
    for res in discovery_result.resources:
        doc = ResearchDocument(
            resource=res,
            text="",
            last_updated=res.last_updated,
            popularity=res.popularity,
        )
        scored_doc = score_document(doc, now=current_time)
        scored_res = scored_doc.resource

        # 2. Check uncertainty condition 1: date is unknown
        if scored_res.last_updated is None:
            scored_res.uncertainty_flag = True
            flagged_uncertainties.append(
                f"Resource '{scored_res.title}' has an unknown publication/update date."
            )

        # 3. Check uncertainty condition 2: stale for its type
        hl = get_half_life_days(scored_res.type)
        is_stale = False
        if scored_res.last_updated is not None:
            ts_updated = (
                scored_res.last_updated
                if scored_res.last_updated.tzinfo is not None
                else scored_res.last_updated.replace(tzinfo=UTC)
            )
            age_seconds = max(0.0, (current_time - ts_updated).total_seconds())
            age_days = age_seconds / 86400.0
            if age_days > hl or scored_res.freshness_score < freshness_threshold:
                is_stale = True
        elif scored_res.freshness_score < freshness_threshold:
            is_stale = True

        if is_stale:
            scored_res.uncertainty_flag = True
            flagged_uncertainties.append(
                f"Resource '{scored_res.title}' is stale for type '{scored_res.type.value}' "
                f"(freshness {scored_res.freshness_score:.2f} < {freshness_threshold})."
            )

        reviewed_resources.append(scored_res)

    # 4. Check uncertainty condition 3: only source for a topic
    topic_counts: dict[str, int] = {}
    for r in reviewed_resources:
        if r.topic and r.topic.strip():
            t = r.topic.strip()
            topic_counts[t] = topic_counts.get(t, 0) + 1

    for r in reviewed_resources:
        if r.topic and r.topic.strip():
            t = r.topic.strip()
            if topic_counts.get(t, 0) == 1:
                r.uncertainty_flag = True
                flagged_uncertainties.append(
                    f"Resource '{r.title}' is the only source for topic '{t}'."
                )

    # 5. Sufficiency checks (min distinct domains, min distinct source types, official/repo)
    distinct_domains = {
        extract_domain(r.url) for r in reviewed_resources if extract_domain(r.url)
    }
    distinct_source_types = {r.type for r in reviewed_resources}

    has_official_or_repo = any(
        r.type == ResourceType.REPO or is_official_docs_domain(r.url)
        for r in reviewed_resources
    )

    domains_ok = len(distinct_domains) >= min_distinct_domains
    types_ok = len(distinct_source_types) >= min_distinct_source_types
    official_or_repo_ok = has_official_or_repo if require_official_or_repo else True
    min_count_ok = len(reviewed_resources) >= 2

    if not domains_ok:
        flagged_uncertainties.append(
            f"Insufficient domain diversity: found {len(distinct_domains)} distinct domain(s), "
            f"minimum required is {min_distinct_domains}."
        )
    if not types_ok:
        flagged_uncertainties.append(
            f"Insufficient perspective diversity: found {len(distinct_source_types)} source type(s), "
            f"minimum required is {min_distinct_source_types}."
        )
    if not official_or_repo_ok:
        flagged_uncertainties.append(
            "Missing authoritative official documentation or repository source."
        )
    if not min_count_ok:
        flagged_uncertainties.append(
            f"Insufficient total resources: found {len(reviewed_resources)}, minimum required is 2."
        )

    is_sufficient = (
        min_count_ok
        and domains_ok
        and types_ok
        and official_or_repo_ok
    )

    is_fresh = (
        all(not r.uncertainty_flag for r in reviewed_resources)
        if reviewed_resources
        else False
    )

    # 6. Evaluate feedback loop trigger: if insufficient and loop_count < MAX_EVIDENCE_RETRY_LOOPS
    needs_more_evidence = (not is_sufficient or not is_fresh) and (loop_count < max_loops)

    # Collect seen URLs for orchestrator retry loop with Stage 3 exclude_urls
    seen_urls: list[str] = [r.url for r in discovery_result.resources if r.url]

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage4_evidence_review",
        goal_id=goal_id,
        is_sufficient=is_sufficient,
        is_fresh=is_fresh,
        needs_more_evidence=needs_more_evidence,
        loop_count=loop_count,
        distinct_domains_count=len(distinct_domains),
        distinct_types_count=len(distinct_source_types),
        has_official_or_repo=has_official_or_repo,
        flagged_count=len(flagged_uncertainties),
    )

    return EvidenceReviewResult(
        goal_id=goal_id,
        is_sufficient=is_sufficient,
        is_fresh=is_fresh,
        reviewed_resources=reviewed_resources,
        flagged_uncertainties=flagged_uncertainties,
        needs_more_evidence=needs_more_evidence,
        loop_count=loop_count,
        seen_urls=seen_urls,
        metrics=metrics,
    )
