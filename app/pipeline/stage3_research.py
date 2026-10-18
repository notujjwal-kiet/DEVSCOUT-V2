"""
Pipeline Stage 3 – Research & Source Discovery.

Responsibility: Discovers high-quality learning resources across multiple sources
(official docs, GitHub repos, engineering blogs, community threads, papers)
to provide diverse, authoritative technical perspectives.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Sequence

import httpx
import structlog

from app.models.pipeline import (
    IntentPlanResult,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
    StageMetrics,
)
from app.research.discovery import discover_sources, normalize_url
from app.research.github import GitHubConnector
from app.research.scoring import hash_content
from app.research.web_fetch import fetch_page

logger = structlog.get_logger(__name__)

STAGE_ID = 3
STAGE_NAME = "Research & Source Discovery"

DEFAULT_STAGE3_CONCURRENCY = 4

# Query templates per topic/milestone
DOCS_QUERY_TEMPLATE = "{topic} official documentation guide"
GITHUB_QUERY_TEMPLATE = "{topic} github repository example code"
COMMUNITY_QUERY_TEMPLATE = "{topic} community discussion best practices gotchas"


@dataclass
class _CandidateSource:
    url: str
    resource_type: ResourceType
    title: str = ""
    topic: str = ""


def _extract_topics(plan_result: IntentPlanResult) -> list[str]:
    """Extract distinct topic and milestone titles from the learning plan."""
    topics: list[str] = []
    if plan_result.plan and plan_result.plan.milestones:
        for m in plan_result.plan.milestones:
            title = m.title.strip()
            if title and title not in topics:
                topics.append(title)

    if not topics and plan_result.plan and plan_result.plan.knowledge_gaps:
        for gap in plan_result.plan.knowledge_gaps:
            clean_gap = gap.strip()
            if clean_gap and clean_gap not in topics:
                topics.append(clean_gap)

    if not topics:
        goal_id = plan_result.plan.goal_id if plan_result.plan else "developer goal"
        topics.append(goal_id)

    return topics


async def run(
    plan_result: IntentPlanResult,
    loop_count: int = 0,
    exclude_urls: Sequence[str] | set[str] | None = None,
    provider: object | None = None,
    github_connector: GitHubConnector | None = None,
    http_client: httpx.AsyncClient | None = None,
    concurrency: int = DEFAULT_STAGE3_CONCURRENCY,
) -> SourceDiscoveryResult:
    """
    Stage 3 entry point: Discover multiple authoritative sources for the learning plan.
    - For each topic/milestone in plan_result, builds 3 query strings (docs, GitHub examples, community) by template.
    - Runs discover_sources for each concurrently.
    - Adds GitHubConnector results.
    - Merges and dedupes by normalized URL, skipping any URLs passed in exclude_urls.
    - Fetches content with fetch_page, builds ResearchDocument -> Resource using hash_content from scoring.py.
    - Drops sources whose fetch fails and logs counts per step.
    - Keeps SourceDiscoveryResult as return type without scoring (Stage 4 scores).
    """
    started_at = datetime.now(UTC)
    goal_id = plan_result.plan.goal_id if plan_result.plan else "unknown-goal"

    topics = _extract_topics(plan_result)

    # 1. Build 3 query strings (docs, GitHub examples, community) by template for each topic
    discovery_queries: list[tuple[str, ResourceType, str]] = []
    for topic in topics:
        discovery_queries.append((DOCS_QUERY_TEMPLATE.format(topic=topic), ResourceType.DOCS, topic))
        discovery_queries.append((GITHUB_QUERY_TEMPLATE.format(topic=topic), ResourceType.REPO, topic))
        discovery_queries.append((COMMUNITY_QUERY_TEMPLATE.format(topic=topic), ResourceType.COMMUNITY, topic))

    owns_client = http_client is None
    active_client = http_client if http_client is not None else httpx.AsyncClient(timeout=10.0)

    try:
        # Initialize GitHubConnector if not provided
        gh = github_connector if github_connector is not None else GitHubConnector(client=active_client)

        discovery_sem = asyncio.Semaphore(concurrency)

        async def _run_discover(q: str, rtype: ResourceType, topic: str) -> list[_CandidateSource]:
            async with discovery_sem:
                try:
                    discovered = await discover_sources(
                        query=q,
                        max_results=5,
                        provider=provider,  # type: ignore[arg-type]
                        http_client=active_client,
                    )
                    return [
                        _CandidateSource(
                            url=ds.url,
                            resource_type=ds.source_type_hint or rtype,
                            title=ds.title,
                            topic=topic,
                        )
                        for ds in discovered
                    ]
                except Exception as exc:
                    logger.warning("stage3_discover_sources_failed", query=q, error=str(exc))
                    return []

        async def _run_github(topic: str) -> list[_CandidateSource]:
            async with discovery_sem:
                try:
                    gh_docs = await gh.search_repositories(
                        query=f"{topic} examples",
                        limit=3,
                    )
                    return [
                        _CandidateSource(
                            url=doc.resource.url,
                            resource_type=ResourceType.REPO,
                            title=doc.resource.title,
                            topic=topic,
                        )
                        for doc in gh_docs
                    ]
                except Exception as exc:
                    logger.warning("stage3_github_search_failed", topic=topic, error=str(exc))
                    return []

        # 2. Run discovery and GitHub queries concurrently bounded by semaphore
        discovery_tasks = [_run_discover(q, rtype, topic) for q, rtype, topic in discovery_queries]
        github_tasks = [_run_github(topic) for topic in topics]
        all_gathered = await asyncio.gather(*discovery_tasks, *github_tasks)

        raw_candidates: list[_CandidateSource] = []
        for batch in all_gathered:
            raw_candidates.extend(batch)

        logger.info(
            "stage3_step1_sources_discovered",
            goal_id=goal_id,
            topics_count=len(topics),
            queries_count=len(discovery_queries) + len(topics),
            raw_discovered_count=len(raw_candidates),
        )

        # 3. Merge and dedupe by normalized URL, skipping exclude_urls
        excluded_normalized: set[str] = {
            normalize_url(u) for u in (exclude_urls or []) if u and u.strip()
        }

        seen_normalized: set[str] = set(excluded_normalized)
        unique_candidates: list[_CandidateSource] = []

        for candidate in raw_candidates:
            if not candidate.url or not candidate.url.strip():
                continue
            norm = normalize_url(candidate.url)
            if not norm or norm in seen_normalized:
                continue
            seen_normalized.add(norm)
            unique_candidates.append(candidate)

        logger.info(
            "stage3_step2_sources_deduped",
            goal_id=goal_id,
            raw_count=len(raw_candidates),
            excluded_count=len(excluded_normalized),
            candidate_count=len(unique_candidates),
        )

        # 4. Fetch content with fetch_page, build ResearchDocument -> Resource with hash_content
        fetch_sem = asyncio.Semaphore(concurrency)

        async def _fetch_one(item: _CandidateSource) -> Resource | None:
            async with fetch_sem:
                try:
                    doc = await fetch_page(
                        url=item.url,
                        resource_type=item.resource_type,
                        client=active_client,
                    )
                    res = doc.resource
                    res.content_hash = hash_content(doc.text)
                    res.last_updated = doc.last_updated
                    res.popularity = doc.popularity
                    res.topic = item.topic
                    # Do NOT score here; Stage 4 handles scoring
                    res.reliability = 0.0
                    res.freshness_score = 0.0
                    res.uncertainty_flag = False
                    return res
                except Exception as exc:
                    logger.warning(
                        "stage3_fetch_failed_dropping_source",
                        url=item.url,
                        error=str(exc),
                    )
                    return None

        fetched_results = await asyncio.gather(*[_fetch_one(c) for c in unique_candidates])

        resources: list[Resource] = [r for r in fetched_results if r is not None]
        failed_fetch_count = len(unique_candidates) - len(resources)

        logger.info(
            "stage3_step3_fetch_completed",
            goal_id=goal_id,
            total_candidates=len(unique_candidates),
            fetched_count=len(resources),
            dropped_failed_count=failed_fetch_count,
        )
    finally:
        if owns_client:
            await active_client.aclose()

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage3_completed",
        goal_id=goal_id,
        resource_count=len(resources),
        loop_count=loop_count,
    )

    return SourceDiscoveryResult(
        goal_id=goal_id,
        resources=resources,
        loop_count=loop_count,
        metrics=metrics,
    )
