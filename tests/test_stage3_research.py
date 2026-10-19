"""
Tests for app/pipeline/stage3_research.py.

Validates:
1. Concurrency cap enforcement for discovery, GitHub searches, and page fetches.
2. Query string generation by template (docs, GitHub examples, community) per topic.
3. Merging and deduplication of candidate sources by normalized URL.
4. Filtering of already-visited URLs via exclude_urls (including normalized forms).
5. Partial fetch failure handling (dropping failed sources without aborting).
6. Resource building with hash_content from scoring.py and unscored contract (Stage 4 scores).
7. Return type preservation (SourceDiscoveryResult).
"""
import asyncio
from datetime import UTC, datetime

import pytest

from app.models.pipeline import (
    ExperienceLevel,
    IntentPlanResult,
    LearningPlan,
    Milestone,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
)
from app.research.github import GitHubConnector
from app.research.models import DiscoveredSource, ResearchDocument
from app.research.scoring import hash_content
from app.pipeline import stage3_research


def _make_plan_result(
    milestone_titles: list[str],
    goal_id: str = "goal-rate-limiter",
) -> IntentPlanResult:
    """Helper to construct an IntentPlanResult with specified milestone titles."""
    milestones = [
        Milestone(
            title=title,
            description=f"Learn and implement {title}",
            target_outcome=f"Understand and implement {title}",
        )
        for title in milestone_titles
    ]
    plan = LearningPlan(
        goal_id=goal_id,
        milestones=milestones,
    )
    return IntentPlanResult(
        goal_id=goal_id,
        plan=plan,
    )


@pytest.mark.asyncio
async def test_stage3_query_generation_and_templates(monkeypatch):
    """Stage 3 generates 3 query strings (docs, github, community) per topic and calls GitHub."""
    captured_discovery_queries: list[str] = []
    captured_github_topics: list[str] = []

    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        captured_discovery_queries.append(query)
        return []

    async def mock_search_repos(self, query, limit=3):
        captured_github_topics.append(query)
        return []

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)

    plan_result = _make_plan_result(["Token Bucket", "Sliding Window"])
    result = await stage3_research.run(plan_result)

    assert isinstance(result, SourceDiscoveryResult)
    assert result.goal_id == "goal-rate-limiter"

    # 2 topics * 3 templates = 6 discovery queries
    assert len(captured_discovery_queries) == 6
    assert "Token Bucket official documentation guide" in captured_discovery_queries
    assert "Token Bucket github repository example code" in captured_discovery_queries
    assert "Token Bucket community discussion best practices gotchas" in captured_discovery_queries
    assert "Sliding Window official documentation guide" in captured_discovery_queries
    assert "Sliding Window github repository example code" in captured_discovery_queries
    assert "Sliding Window community discussion best practices gotchas" in captured_discovery_queries

    # 2 topics * 1 github query
    assert len(captured_github_topics) == 2
    assert "Token Bucket examples" in captured_github_topics
    assert "Sliding Window examples" in captured_github_topics


@pytest.mark.asyncio
async def test_stage3_dedupe_normalized_urls(monkeypatch):
    """Duplicate URLs differing by utm params, trailing slash, or fragments are deduped."""
    fetched_urls: list[str] = []

    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        # Return URLs that normalize to the same canonical URL
        return [
            DiscoveredSource(
                url="https://docs.python.org/3/library/asyncio.html?utm_source=twitter#intro",
                title="Python Asyncio",
                domain="docs.python.org",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://docs.python.org/3/library/asyncio.html",
                title="Python Asyncio Docs",
                domain="docs.python.org",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://example.com/guide/",
                title="Guide Trailing Slash",
                domain="example.com",
                source_type_hint=ResourceType.BLOG,
            ),
            DiscoveredSource(
                url="https://example.com/guide?utm_medium=email",
                title="Guide Clean",
                domain="example.com",
                source_type_hint=ResourceType.BLOG,
            ),
        ]

    async def mock_search_repos(self, query, limit=3):
        return [
            ResearchDocument(
                resource=Resource(
                    url="https://example.com/guide",  # Duplicate of the blog above
                    type=ResourceType.REPO,
                    title="Repo Guide",
                    content_hash="repohash",
                ),
                text="Repo guide readme",
            )
        ]

    async def mock_fetch_page(url, resource_type, client=None):
        fetched_urls.append(url)
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title=f"Page {url}",
                content_hash="initial_hash",
                retrieved_at=datetime.now(UTC),
            ),
            text=f"Sample body text for {url}",
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    plan_result = _make_plan_result(["Async Patterns"])
    result = await stage3_research.run(plan_result)

    # Only 2 distinct normalized URLs should be fetched:
    # 1. https://docs.python.org/3/library/asyncio.html
    # 2. https://example.com/guide
    assert len(fetched_urls) == 2
    assert len(result.resources) == 2

    urls_in_result = [r.url for r in result.resources]
    assert "https://docs.python.org/3/library/asyncio.html?utm_source=twitter#intro" in urls_in_result
    assert "https://example.com/guide/" in urls_in_result


@pytest.mark.asyncio
async def test_stage3_exclude_urls(monkeypatch):
    """URLs matching exclude_urls (even with differing utm params/trailing slash) are skipped."""
    fetched_urls: list[str] = []

    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://docs.python.org/3/library/asyncio.html",
                title="Python Asyncio",
                domain="docs.python.org",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://fastapi.tiangolo.com/tutorial/",
                title="FastAPI Tutorial",
                domain="fastapi.tiangolo.com",
                source_type_hint=ResourceType.DOCS,
            ),
        ]

    async def mock_search_repos(self, query, limit=3):
        return []

    async def mock_fetch_page(url, resource_type, client=None):
        fetched_urls.append(url)
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title=f"Page {url}",
                content_hash="h1",
            ),
            text="Content",
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    # Exclude asyncio.html but with utm and anchor in the exclude set
    exclude_list = ["https://docs.python.org/3/library/asyncio.html?utm_campaign=loop#part1"]

    plan_result = _make_plan_result(["FastAPI Async"])
    result = await stage3_research.run(plan_result, exclude_urls=exclude_list)

    # Only FastAPI tutorial should be fetched and returned
    assert len(fetched_urls) == 1
    assert fetched_urls[0] == "https://fastapi.tiangolo.com/tutorial/"
    assert len(result.resources) == 1
    assert result.resources[0].url == "https://fastapi.tiangolo.com/tutorial/"


@pytest.mark.asyncio
async def test_stage3_partial_fetch_failure(monkeypatch):
    """Sources whose fetch fails are dropped, while successful ones are retained."""
    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://reliable.example.com/docs",
                title="Reliable Docs",
                domain="reliable.example.com",
                source_type_hint=ResourceType.DOCS,
            ),
            DiscoveredSource(
                url="https://broken.example.com/broken-page",
                title="Broken Page",
                domain="broken.example.com",
                source_type_hint=ResourceType.BLOG,
            ),
        ]

    async def mock_search_repos(self, query, limit=3):
        return [
            ResearchDocument(
                resource=Resource(
                    url="https://github.com/example/good-repo",
                    type=ResourceType.REPO,
                    title="Good Repo",
                    content_hash="repohash",
                ),
                text="Repo readme content",
            )
        ]

    async def mock_fetch_page(url, resource_type, client=None):
        if "broken" in url:
            raise RuntimeError(f"Simulated network timeout fetching {url}")

        body = f"Fetched content for {url} with substantial educational material."
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title=f"Title for {url}",
                content_hash="initial",
            ),
            text=body,
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    plan_result = _make_plan_result(["Resilience"])
    result = await stage3_research.run(plan_result)

    assert isinstance(result, SourceDiscoveryResult)
    # The broken URL was dropped; the 2 good ones remain
    assert len(result.resources) == 2
    resource_urls = [r.url for r in result.resources]
    assert "https://reliable.example.com/docs" in resource_urls
    assert "https://github.com/example/good-repo" in resource_urls
    assert "https://broken.example.com/broken-page" not in resource_urls

    # Check that content_hash is computed via hash_content
    for r in result.resources:
        expected_body = f"Fetched content for {r.url} with substantial educational material."
        assert r.content_hash == hash_content(expected_body)


@pytest.mark.asyncio
async def test_stage3_concurrency_cap(monkeypatch):
    """Concurrency during fetch does not exceed the specified semaphore limit."""
    active_fetches = 0
    max_observed_fetches = 0
    lock = asyncio.Lock()

    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url=f"https://example.com/page{i}",
                title=f"Page {i}",
                domain="example.com",
                source_type_hint=ResourceType.DOCS,
            )
            for i in range(8)
        ]

    async def mock_search_repos(self, query, limit=3):
        return []

    async def mock_fetch_page(url, resource_type, client=None):
        nonlocal active_fetches, max_observed_fetches
        async with lock:
            active_fetches += 1
            if active_fetches > max_observed_fetches:
                max_observed_fetches = active_fetches

        # Sleep briefly to give concurrent tasks an opportunity to overlap
        await asyncio.sleep(0.02)

        async with lock:
            active_fetches -= 1

        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title=f"Title {url}",
                content_hash="h",
            ),
            text=f"Body for {url}",
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    plan_result = _make_plan_result(["Concurrency"])
    concurrency_limit = 2
    result = await stage3_research.run(plan_result, concurrency=concurrency_limit)

    assert len(result.resources) == 8
    # Assert active concurrency never exceeded 2
    assert max_observed_fetches <= concurrency_limit


@pytest.mark.asyncio
async def test_stage3_unscored_contract(monkeypatch):
    """Stage 3 leaves resources unscored (reliability=0.0, freshness_score=0.0) for Stage 4."""
    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://example.com/unscored",
                title="Unscored Page",
                domain="example.com",
                source_type_hint=ResourceType.DOCS,
            )
        ]

    async def mock_search_repos(self, query, limit=3):
        return []

    async def mock_fetch_page(url, resource_type, client=None):
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title="Unscored Page",
                content_hash="hash000",
                reliability=0.99,  # Even if fetch_page had some value
                freshness_score=0.99,
            ),
            text="Unscored text",
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    plan_result = _make_plan_result(["Contract"])
    result = await stage3_research.run(plan_result)

    assert len(result.resources) == 1
    res = result.resources[0]
    # Do NOT score here; Stage 4 handles scoring
    assert res.reliability == 0.0
    assert res.freshness_score == 0.0
    assert res.uncertainty_flag is False
    assert res.content_hash == hash_content("Unscored text")


@pytest.mark.asyncio
async def test_stage3_stores_cleaned_excerpt_and_preserves_hash(monkeypatch):
    """Stage 3 stores a cleaned excerpt of doc.text and preserves content_hash."""
    from app.pipeline.stage3_research import extract_clean_excerpt

    # Test extract_clean_excerpt helper directly
    raw_nav_text = """
    Home > Docs > Caching
    Skip to main content
    
    Distributed caching allows systems to store frequently accessed data in high-speed storage.
    By distributing data across multiple cache nodes, applications can achieve horizontal scalability.
    
    Copyright 2026 Example Corp. All rights reserved.
    """
    cleaned = extract_clean_excerpt(raw_nav_text, max_chars=100)
    assert not cleaned.startswith("Home")
    assert not cleaned.startswith("Skip to main content")
    assert "Distributed caching" in cleaned
    assert len(cleaned) <= 100

    # Test in Stage 3 run()
    full_body = (
        "Distributed caching fundamentals: memory caches like Redis store key-value pairs "
        "in RAM for sub-millisecond retrieval. Consistent hashing distributes keys evenly."
    )

    async def mock_discover_sources(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://example.com/caching-guide",
                title="Caching Guide",
                domain="example.com",
                source_type_hint=ResourceType.DOCS,
            )
        ]

    async def mock_search_repos(self, query, limit=3):
        return []

    async def mock_fetch_page(url, resource_type, client=None):
        return ResearchDocument(
            resource=Resource(
                url=url,
                type=resource_type,
                title="Caching Guide",
                content_hash="raw_hash",
            ),
            text=full_body,
        )

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", mock_discover_sources)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", mock_search_repos)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", mock_fetch_page)

    plan_result = _make_plan_result(["Caching"])
    result = await stage3_research.run(plan_result)

    assert len(result.resources) == 1
    resource = result.resources[0]
    # content_hash is intact and matches full text hash
    assert resource.content_hash == hash_content(full_body)
    # excerpt is populated with cleaned text
    assert "Distributed caching fundamentals" in resource.excerpt
    assert resource.excerpt != ""

