"""
Pipeline Stage 3 – Research & Source Discovery.

Responsibility: Discovers high-quality learning resources across multiple sources
(official docs, GitHub repos, engineering blogs, community threads, papers)
to provide diverse, authoritative technical perspectives.
"""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime
import structlog

from app.models.pipeline import (
    ExperienceLevel,
    IntentPlanResult,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
    StageMetrics,
)

logger = structlog.get_logger(__name__)

STAGE_ID = 3
STAGE_NAME = "Research & Source Discovery"


async def run(
    plan_result: IntentPlanResult,
    loop_count: int = 0,
) -> SourceDiscoveryResult:
    """
    Stage 3 entry point: Discover multiple authoritative sources for learning plan.
    """
    started_at = datetime.now(UTC)
    goal_id = plan_result.plan.goal_id

    # Stub: Curate sources across multiple perspectives
    sources_data = [
        ("https://docs.python.org/3/", ResourceType.DOCS, "Official Documentation", 0.95, 0.90),
        ("https://github.com/fastapi/fastapi", ResourceType.REPO, "Reference Repository", 0.90, 0.85),
        ("https://netflixtechblog.com", ResourceType.BLOG, "Architecture Case Study", 0.85, 0.80),
        ("https://stackoverflow.com/questions", ResourceType.COMMUNITY, "Community Discussion & Gotchas", 0.75, 0.85),
    ]

    resources: list[Resource] = []
    for url, rtype, title, rel, fresh in sources_data:
        content_hash = hashlib.sha256(f"{url}:{title}".encode()).hexdigest()[:16]
        resources.append(
            Resource(
                url=url,
                type=rtype,
                title=title,
                content_hash=content_hash,
                reliability=rel,
                freshness_score=fresh,
                level_fit=ExperienceLevel.INTERMEDIATE,
                uncertainty_flag=False,
                summary=f"Curated reference for {title}.",
            )
        )

    ended_at = datetime.now(UTC)
    metrics = StageMetrics(
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=(ended_at - started_at).total_seconds() * 1000.0,
    )

    logger.info(
        "stage3_source_discovery",
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
