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
import re
from typing import Sequence


import httpx
import structlog

from app.config import get_settings
from app.llm.provider import RateLimitError
from app.models.pipeline import (
    IntentPlanResult,
    LearningGoal,
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


BOILERPLATE_PATTERNS = [
    re.compile(r"^skip to\b", re.I),
    re.compile(r"^home\s*>", re.I),
    re.compile(r"cookie policy|privacy policy", re.I),
    re.compile(r"copyright\s+(\d{4})?", re.I),
    re.compile(r"all rights reserved", re.I),
]


def extract_clean_excerpt(text: str, max_chars: int | None = None) -> str:
    """
    Extract a cleaned excerpt of the first relevant content chunk:
    - Drops raw navigation and boilerplate lines (e.g. skip links, breadcrumbs, copyright).
    - Normalizes extra whitespace and captures substantive content up to max_chars characters.
    """
    if not text:
        return ""

    limit = max_chars if max_chars is not None else get_settings().max_excerpt_chars

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned_chunks: list[str] = []

    for line in lines:
        if any(pattern.search(line) for pattern in BOILERPLATE_PATTERNS):
            continue
        cleaned_chunks.append(line)

    combined = " ".join(" ".join(cleaned_chunks).split()).strip()
    if not combined:
        combined = " ".join(text.split()).strip()

    return combined[:limit].strip()




@dataclass
class _CandidateSource:
    url: str
    resource_type: ResourceType
    title: str = ""
    topic: str = ""


def extract_topics(plan_result: IntentPlanResult) -> list[str]:
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


_extract_topics = extract_topics


async def run(
    plan_result: IntentPlanResult,
    loop_count: int = 0,
    exclude_urls: Sequence[str] | set[str] | None = None,
    only_topics: Sequence[str] | None = None,
    provider: object | None = None,
    github_connector: GitHubConnector | None = None,
    http_client: httpx.AsyncClient | None = None,
    concurrency: int = DEFAULT_STAGE3_CONCURRENCY,
    tech_stack: Sequence[str] | None = None,
    goal: str | LearningGoal | None = None,
    enable_github_queries: bool | None = None,
    enable_community_queries: bool | None = None,
) -> SourceDiscoveryResult:
    """
    Stage 3 entry point: Discover multiple authoritative sources for the learning plan.
    - For each topic/milestone in plan_result, builds 3 query strings (docs, GitHub examples, community) by template.
    - If only_topics is provided, retries search only those topics.
    - Runs discover_sources for each concurrently.
    - Adds GitHubConnector results.
    - Merges and dedupes by normalized URL, skipping any URLs passed in exclude_urls.
    - Fetches content with fetch_page, builds ResearchDocument -> Resource using hash_content from scoring.py.
    - Drops sources whose fetch fails and logs counts per step.
    - Keeps SourceDiscoveryResult as return type without scoring (Stage 4 scores).
    """
    cfg = get_settings()
    started_at = datetime.now(UTC)
    goal_id = plan_result.plan.goal_id if plan_result.plan else "unknown-goal"

    all_topics = extract_topics(plan_result)
    if only_topics is not None:
        clean_only = [t.strip() for t in only_topics if t and t.strip()]
        only_lookup = {t.lower() for t in clean_only}
        topics = [t for t in all_topics if t.lower() in only_lookup]
        for t in clean_only:
            if t.lower() not in {x.lower() for x in topics}:
                topics.append(t)
    else:
        topics = all_topics

    # Enforce Settings cap on topics per run
    if len(topics) > cfg.max_topics_per_run:
        topics = topics[:cfg.max_topics_per_run]

    # Build context prefix from tech_stack and goal
    context_parts: list[str] = []
    if tech_stack:
        for t in tech_stack:
            clean_t = t.strip()
            if clean_t and clean_t not in context_parts:
                context_parts.append(clean_t)
    if goal:
        goal_text = goal.title if isinstance(goal, LearningGoal) else str(goal).strip()
        if goal_text and not goal_text.startswith("goal-"):
            clean_g = goal_text.split("\n")[0][:60].strip()
            for w in clean_g.split():
                if w.lower() not in [p.lower() for p in context_parts]:
                    context_parts.append(w)
    context_prefix = " ".join(context_parts).strip()

    # Determine language from tech_stack for GitHub queries
    lang: str | None = None
    if tech_stack:
        for t in tech_stack:
            tl = t.strip().lower()
            if tl in {"python", "javascript", "typescript", "go", "rust", "java", "cpp", "c#", "ruby", "php"}:
                lang = tl
                break
            elif tl in {"fastapi", "django", "flask"}:
                lang = "python"
                break
            elif tl in {"react", "node", "express", "vue", "nextjs"}:
                lang = "javascript"
                break

    allow_gh_queries = enable_github_queries if enable_github_queries is not None else cfg.enable_github_queries
    allow_comm_queries = enable_community_queries if enable_community_queries is not None else cfg.enable_community_queries

    # 1. Build query strings for each topic (docs by default; github & community optional/off by default)
    discovery_queries: list[tuple[str, ResourceType, str]] = []
    for topic in topics:
        if context_prefix:
            discovery_queries.append((f"{context_prefix} {topic} official documentation", ResourceType.DOCS, topic))
            if allow_gh_queries:
                discovery_queries.append((f"{context_prefix} {topic} github repository example code stars:>50", ResourceType.REPO, topic))
            if allow_comm_queries:
                discovery_queries.append((f"{context_prefix} {topic} community discussion best practices gotchas", ResourceType.COMMUNITY, topic))
        else:
            discovery_queries.append((DOCS_QUERY_TEMPLATE.format(topic=topic), ResourceType.DOCS, topic))
            if allow_gh_queries:
                discovery_queries.append((GITHUB_QUERY_TEMPLATE.format(topic=topic), ResourceType.REPO, topic))
            if allow_comm_queries:
                discovery_queries.append((COMMUNITY_QUERY_TEMPLATE.format(topic=topic), ResourceType.COMMUNITY, topic))

    owns_client = http_client is None
    active_client = http_client if http_client is not None else httpx.AsyncClient(timeout=10.0)

    rate_limited_searches = 0
    discovery_errors: list[str] = []

    try:
        # Initialize GitHubConnector if not provided
        gh = github_connector if github_connector is not None else GitHubConnector(client=active_client)

        discovery_sem = asyncio.Semaphore(concurrency)

        async def _run_discover(q: str, rtype: ResourceType, topic: str) -> list[_CandidateSource]:
            nonlocal rate_limited_searches
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
                except RateLimitError as exc:
                    rate_limited_searches += 1
                    err_msg = f"Rate limit during web discovery for query '{q}': {exc}"
                    discovery_errors.append(err_msg)
                    logger.warning("stage3_discover_sources_rate_limited", query=q, error=str(exc))
                    return []
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "429" in exc_str or "quota" in exc_str or "rate limit" in exc_str:
                        rate_limited_searches += 1
                        err_msg = f"Rate limit during web discovery for query '{q}': {exc}"
                        discovery_errors.append(err_msg)
                        logger.warning("stage3_discover_sources_rate_limited", query=q, error=str(exc))
                        return []
                    err_msg = f"Error during web discovery for query '{q}': {exc}"
                    discovery_errors.append(err_msg)
                    logger.warning("stage3_discover_sources_failed", query=q, error=str(exc))
                    return []

        async def _run_github(topic: str) -> list[_CandidateSource]:
            nonlocal rate_limited_searches
            async with discovery_sem:
                try:
                    if context_prefix or lang:
                        gh_q = f"{context_prefix} {topic} examples".strip()
                        if lang:
                            gh_q += f" language:{lang}"
                        gh_q += " stars:>50"
                    else:
                        gh_q = f"{topic} examples"

                    gh_docs = await gh.search_repositories(
                        query=gh_q,
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
                except RateLimitError as exc:
                    rate_limited_searches += 1
                    err_msg = f"Rate limit during GitHub search for topic '{topic}': {exc}"
                    discovery_errors.append(err_msg)
                    logger.warning("stage3_github_search_rate_limited", topic=topic, error=str(exc))
                    return []
                except Exception as exc:
                    exc_str = str(exc).lower()
                    if "429" in exc_str or "rate limit" in exc_str:
                        rate_limited_searches += 1
                        err_msg = f"Rate limit during GitHub search for topic '{topic}': {exc}"
                        discovery_errors.append(err_msg)
                        logger.warning("stage3_github_search_rate_limited", topic=topic, error=str(exc))
                        return []
                    err_msg = f"Error during GitHub search for topic '{topic}': {exc}"
                    discovery_errors.append(err_msg)
                    logger.warning("stage3_github_search_failed", topic=topic, error=str(exc))
                    return []

        # 2. Schedule discovery and GitHub queries bounded by max_searches_per_run
        max_searches = cfg.max_searches_per_run
        all_tasks = []
        for q, rtype, topic in discovery_queries:
            if len(all_tasks) < max_searches:
                all_tasks.append(_run_discover(q, rtype, topic))
        for topic in topics:
            if len(all_tasks) < max_searches:
                all_tasks.append(_run_github(topic))

        all_gathered = await asyncio.gather(*all_tasks)

        raw_candidates: list[_CandidateSource] = []
        for batch in all_gathered:
            raw_candidates.extend(batch)

        logger.info(
            "stage3_step1_sources_discovered",
            goal_id=goal_id,
            topics_count=len(topics),
            queries_count=len(all_tasks),
            raw_discovered_count=len(raw_candidates),
            rate_limited_searches=rate_limited_searches,
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
                    res.excerpt = extract_clean_excerpt(doc.text, max_chars=cfg.max_excerpt_chars)
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
                    err_msg = f"Failed to fetch content from '{item.url}': {exc}"
                    discovery_errors.append(err_msg)
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
        rate_limited_searches=rate_limited_searches,
        degraded=rate_limited_searches > 0,
    )

    return SourceDiscoveryResult(
        goal_id=goal_id,
        resources=resources,
        discovered_count=len(raw_candidates),
        fetched_count=len(resources),
        errors=discovery_errors,
        loop_count=loop_count,
        degraded=rate_limited_searches > 0,
        rate_limited_searches=rate_limited_searches,
        metrics=metrics,
    )
