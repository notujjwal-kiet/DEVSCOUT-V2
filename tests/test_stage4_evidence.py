"""
Tests for app/pipeline/stage4_evidence_validation.py.

Validates:
1. Stale blog detection (date older than half-life flagged with uncertainty).
2. Unknown date handling (last_updated is None flagged with uncertainty).
3. Single domain insufficiency (distinct domains < min_distinct_domains).
4. Feedback loop cap (loop_count >= MAX_EVIDENCE_RETRY_LOOPS terminates retry).
5. Only source for a topic flagged with uncertainty.
6. Multi-perspective sufficiency (distinct types and official/repo requirement).
7. Return of seen_urls when insufficient for retry loops in Stage 3.
"""
from datetime import UTC, datetime, timedelta
import pytest

from app.models.pipeline import (
    EvidenceReviewResult,
    ExperienceLevel,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
)
from app.pipeline import stage4_evidence_validation


def _make_resource(
    url: str,
    rtype: ResourceType,
    title: str = "Test Resource",
    last_updated: datetime | None = None,
    popularity: int | None = None,
    topic: str | None = None,
) -> Resource:
    """Helper to construct a Resource with metadata."""
    return Resource(
        url=url,
        type=rtype,
        title=title,
        content_hash="test_hash_123",
        last_updated=last_updated,
        popularity=popularity,
        topic=topic,
    )


@pytest.mark.asyncio
async def test_stage4_stale_blog():
    """Blog older than half-life (180 days) is scored via decay and flagged with uncertainty."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    # 250 days old > 180 days half-life for blog
    stale_date = now - timedelta(days=250)

    res_blog = _make_resource(
        url="https://engineering.example.com/rates",
        rtype=ResourceType.BLOG,
        title="Distributed Rate Limiting at Scale",
        last_updated=stale_date,
    )
    res_docs = _make_resource(
        url="https://docs.python.org/3/library/asyncio.html",
        rtype=ResourceType.DOCS,
        title="Asyncio Documentation",
        last_updated=now - timedelta(days=10),
    )

    discovery = SourceDiscoveryResult(
        goal_id="goal-stale-test",
        resources=[res_blog, res_docs],
        loop_count=0,
    )

    result = await stage4_evidence_validation.run(discovery, now=now)

    assert isinstance(result, EvidenceReviewResult)
    # Blog is stale -> freshness < 0.5 and uncertainty_flag=True
    reviewed_blog = next(r for r in result.reviewed_resources if r.url == res_blog.url)
    assert reviewed_blog.freshness_score < 0.5
    assert reviewed_blog.uncertainty_flag is True

    # Fresh docs resource is not flagged
    reviewed_docs = next(r for r in result.reviewed_resources if r.url == res_docs.url)
    assert reviewed_docs.freshness_score > 0.9
    assert reviewed_docs.uncertainty_flag is False

    # Overall is_fresh is False because of the stale blog
    assert result.is_fresh is False
    assert any("stale" in msg.lower() for msg in result.flagged_uncertainties)
    # Since not fresh, needs more evidence
    assert result.needs_more_evidence is True


@pytest.mark.asyncio
async def test_stage4_unknown_date():
    """Resource with unknown date (last_updated=None) receives default freshness and uncertainty flag."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    res_undated = _make_resource(
        url="https://docs.python.org/3/library/time.html",
        rtype=ResourceType.DOCS,
        title="Python Time Module",
        last_updated=None,  # Unknown publication date
    )
    res_repo = _make_resource(
        url="https://github.com/fastapi/fastapi",
        rtype=ResourceType.REPO,
        title="FastAPI Repo",
        last_updated=now - timedelta(days=5),
        popularity=70000,
    )

    discovery = SourceDiscoveryResult(
        goal_id="goal-unknown-date",
        resources=[res_undated, res_repo],
        loop_count=0,
    )

    result = await stage4_evidence_validation.run(discovery, now=now)

    reviewed_undated = next(r for r in result.reviewed_resources if r.url == res_undated.url)
    # Default freshness for None date is 0.3
    assert reviewed_undated.freshness_score == pytest.approx(0.3, rel=1e-2)
    assert reviewed_undated.uncertainty_flag is True

    assert result.is_fresh is False
    assert any("unknown publication/update date" in msg.lower() for msg in result.flagged_uncertainties)


@pytest.mark.asyncio
async def test_stage4_single_domain_insufficient():
    """When all resources belong to a single domain, sufficiency check fails and seen_urls are returned."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    # 3 distinct resources, but all from python.org
    res1 = _make_resource(
        url="https://docs.python.org/3/library/asyncio.html",
        rtype=ResourceType.DOCS,
        title="Asyncio",
        last_updated=now - timedelta(days=1),
    )
    res2 = _make_resource(
        url="https://docs.python.org/3/library/threading.html",
        rtype=ResourceType.DOCS,
        title="Threading",
        last_updated=now - timedelta(days=2),
    )
    res3 = _make_resource(
        url="https://docs.python.org/3/library/multiprocessing.html",
        rtype=ResourceType.DOCS,
        title="Multiprocessing",
        last_updated=now - timedelta(days=3),
    )

    discovery = SourceDiscoveryResult(
        goal_id="goal-single-domain",
        resources=[res1, res2, res3],
        loop_count=0,
    )

    result = await stage4_evidence_validation.run(discovery, now=now, min_distinct_domains=2)

    # Failed domain diversity check
    assert result.is_sufficient is False
    assert result.needs_more_evidence is True
    # seen_urls returned for retry loop exclusion in Stage 3
    assert len(result.seen_urls) == 3
    assert res1.url in result.seen_urls
    assert res2.url in result.seen_urls
    assert res3.url in result.seen_urls
    assert any("domain diversity" in msg.lower() for msg in result.flagged_uncertainties)


@pytest.mark.asyncio
async def test_stage4_loop_cap():
    """When evidence is insufficient but loop_count >= MAX_EVIDENCE_RETRY_LOOPS, retry loop stops."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    # Insufficient evidence (0 resources)
    empty_discovery = SourceDiscoveryResult(
        goal_id="goal-loop-cap",
        resources=[],
        loop_count=2,  # Already at MAX_EVIDENCE_RETRY_LOOPS
    )

    result = await stage4_evidence_validation.run(empty_discovery, now=now, max_loops=2)

    assert result.is_sufficient is False
    # Loop cap reached -> needs_more_evidence must be False!
    assert result.needs_more_evidence is False
    assert result.loop_count == 2


@pytest.mark.asyncio
async def test_stage4_only_source_for_topic():
    """Resource that is the sole source covering a topic is marked with uncertainty_flag."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    # Topic A has 2 sources (well-supported)
    res_a1 = _make_resource(
        url="https://docs.python.org/3/library/asyncio.html",
        rtype=ResourceType.DOCS,
        title="Python Asyncio",
        last_updated=now,
        topic="Concurrency",
    )
    res_a2 = _make_resource(
        url="https://github.com/python/cpython",
        rtype=ResourceType.REPO,
        title="CPython",
        last_updated=now,
        topic="Concurrency",
    )
    # Topic B has only 1 source (under-evidenced singleton topic)
    res_b1 = _make_resource(
        url="https://stackoverflow.com/questions/12345/redis-locks",
        rtype=ResourceType.COMMUNITY,
        title="Redis Locks Gotchas",
        last_updated=now,
        topic="Distributed Locking",
    )

    discovery = SourceDiscoveryResult(
        goal_id="goal-singleton-topic",
        resources=[res_a1, res_a2, res_b1],
        loop_count=0,
    )

    result = await stage4_evidence_validation.run(discovery, now=now)

    reviewed_a1 = next(r for r in result.reviewed_resources if r.url == res_a1.url)
    reviewed_a2 = next(r for r in result.reviewed_resources if r.url == res_a2.url)
    reviewed_b1 = next(r for r in result.reviewed_resources if r.url == res_b1.url)

    # Concurrency has 2 sources -> no singleton uncertainty
    assert reviewed_a1.uncertainty_flag is False
    assert reviewed_a2.uncertainty_flag is False

    # Distributed Locking has only 1 source -> marked uncertainty_flag
    assert reviewed_b1.uncertainty_flag is True
    assert any("only source for topic 'distributed locking'" in msg.lower() for msg in result.flagged_uncertainties)


@pytest.mark.asyncio
async def test_stage4_multi_perspective_sufficiency():
    """All sufficiency checks pass when distinct domains, types, and official sources are present."""
    now = datetime(2026, 10, 18, 12, 0, tzinfo=UTC)

    res_docs = _make_resource(
        url="https://fastapi.tiangolo.com/tutorial/",
        rtype=ResourceType.DOCS,
        title="FastAPI Official Guide",
        last_updated=now - timedelta(days=10),
    )
    res_repo = _make_resource(
        url="https://github.com/tiangolo/fastapi",
        rtype=ResourceType.REPO,
        title="FastAPI Repository",
        last_updated=now - timedelta(days=5),
        popularity=75000,
    )
    res_comm = _make_resource(
        url="https://stackoverflow.com/questions/999/fastapi-concurrency",
        rtype=ResourceType.COMMUNITY,
        title="Community Discussion",
        last_updated=now - timedelta(days=15),
        popularity=120,
    )

    discovery = SourceDiscoveryResult(
        goal_id="goal-sufficient",
        resources=[res_docs, res_repo, res_comm],
        loop_count=0,
    )

    result = await stage4_evidence_validation.run(discovery, now=now)

    # Sufficiency: 3 distinct domains, 3 distinct types, official docs and repo present
    assert result.is_sufficient is True
    assert result.is_fresh is True
    assert result.needs_more_evidence is False
    assert len(result.reviewed_resources) == 3
