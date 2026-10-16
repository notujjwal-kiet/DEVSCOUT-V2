"""
Tests for app/research/github.py – GitHubConnector with mocked httpx.
Covers:
- normal search with README decoding and metadata
- README truncation to 20,000 characters
- skipping repositories without a README (404)
- rate limit handling (403/429) on search and readme
- network error handling mapped to ResearchError
- Authorization token header sent only when token is set
"""
import base64
from datetime import UTC, datetime
import json

import httpx
import pytest

from app.models.pipeline import ResourceType
from app.research.github import GitHubConnector, hash_content
from app.research.models import ResearchDocument, ResearchError


@pytest.mark.asyncio
async def test_github_normal_search():
    """search_repositories discovers repos, decodes READMEs, and builds ResearchDocuments."""
    recorded_requests: list[httpx.Request] = []

    readme1_text = "# Repo One\n\nA distributed token bucket rate limiter."
    readme2_text = "# Repo Two\n\nSliding window counter algorithm in Python."

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_requests.append(request)
        path = request.url.path

        if path == "/search/repositories":
            assert request.url.params["q"] == "rate-limiter"
            assert request.url.params["sort"] == "stars"
            assert request.url.params["order"] == "desc"
            assert request.url.params["per_page"] == "2"
            return httpx.Response(
                200,
                json={
                    "total_count": 2,
                    "items": [
                        {
                            "name": "limiter-one",
                            "full_name": "alpha/limiter-one",
                            "owner": {"login": "alpha"},
                            "html_url": "https://github.com/alpha/limiter-one",
                            "stargazers_count": 1200,
                            "pushed_at": "2026-10-10T14:30:00Z",
                        },
                        {
                            "name": "limiter-two",
                            "full_name": "beta/limiter-two",
                            "owner": {"login": "beta"},
                            "html_url": "https://github.com/beta/limiter-two",
                            "stargazers_count": 450,
                            "pushed_at": "2026-09-05T08:15:00Z",
                        },
                    ],
                },
            )

        if path == "/repos/alpha/limiter-one/readme":
            content_b64 = base64.b64encode(readme1_text.encode("utf-8")).decode("utf-8")
            return httpx.Response(200, json={"content": content_b64, "encoding": "base64"})

        if path == "/repos/beta/limiter-two/readme":
            content_b64 = base64.b64encode(readme2_text.encode("utf-8")).decode("utf-8")
            return httpx.Response(200, json={"content": content_b64, "encoding": "base64"})

        return httpx.Response(404, json={"message": "Not found"})

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    docs = await connector.search_repositories("rate-limiter", limit=2)
    assert len(docs) == 2
    assert all(isinstance(d, ResearchDocument) for d in docs)

    doc1 = docs[0]
    assert doc1.resource.url == "https://github.com/alpha/limiter-one"
    assert doc1.resource.type == ResourceType.REPO
    assert doc1.resource.title == "alpha/limiter-one"
    assert doc1.resource.content_hash == hash_content(readme1_text)
    assert doc1.text == readme1_text
    assert doc1.popularity == 1200
    assert doc1.last_updated == datetime(2026, 10, 10, 14, 30, tzinfo=UTC)

    doc2 = docs[1]
    assert doc2.resource.url == "https://github.com/beta/limiter-two"
    assert doc2.resource.title == "beta/limiter-two"
    assert doc2.resource.content_hash == hash_content(readme2_text)
    assert doc2.text == readme2_text
    assert doc2.popularity == 450
    assert doc2.last_updated == datetime(2026, 9, 5, 8, 15, tzinfo=UTC)


@pytest.mark.asyncio
async def test_github_truncates_long_readme():
    """README text is truncated to 20,000 characters and content_hash reflects truncated text."""
    long_text = "A" * 25000

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/repositories":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "name": "huge-repo",
                            "full_name": "org/huge-repo",
                            "owner": {"login": "org"},
                            "html_url": "https://github.com/org/huge-repo",
                            "stargazers_count": 5000,
                            "pushed_at": "2026-10-01T00:00:00Z",
                        }
                    ]
                },
            )
        if path == "/repos/org/huge-repo/readme":
            content_b64 = base64.b64encode(long_text.encode("utf-8")).decode("utf-8")
            return httpx.Response(200, json={"content": content_b64, "encoding": "base64"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    docs = await connector.search_repositories("huge", limit=1)
    assert len(docs) == 1
    assert len(docs[0].text) == 20000
    assert docs[0].text == "A" * 20000
    assert docs[0].resource.content_hash == hash_content("A" * 20000)


@pytest.mark.asyncio
async def test_github_repo_without_readme_skipped():
    """Repositories with 404 on README are skipped instead of failing."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/repositories":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "name": "repo-with-readme",
                            "full_name": "user/repo-with-readme",
                            "owner": {"login": "user"},
                            "html_url": "https://github.com/user/repo-with-readme",
                            "stargazers_count": 100,
                        },
                        {
                            "name": "repo-no-readme",
                            "full_name": "user/repo-no-readme",
                            "owner": {"login": "user"},
                            "html_url": "https://github.com/user/repo-no-readme",
                            "stargazers_count": 50,
                        },
                    ]
                },
            )
        if path == "/repos/user/repo-with-readme/readme":
            content_b64 = base64.b64encode(b"README content").decode("utf-8")
            return httpx.Response(200, json={"content": content_b64, "encoding": "base64"})
        if path == "/repos/user/repo-no-readme/readme":
            return httpx.Response(404, json={"message": "Not Found"})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    docs = await connector.search_repositories("query", limit=2)
    assert len(docs) == 1
    assert docs[0].resource.title == "user/repo-with-readme"


@pytest.mark.asyncio
async def test_github_rate_limit_error_search():
    """Rate limit error (403/429) on search endpoint raises readable ResearchError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded for user IP"},
        )

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    with pytest.raises(ResearchError, match="rate limit exceeded"):
        await connector.search_repositories("query")


@pytest.mark.asyncio
async def test_github_rate_limit_error_readme():
    """Rate limit error on README fetch raises readable ResearchError."""
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/search/repositories":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "name": "repo",
                            "full_name": "user/repo",
                            "owner": {"login": "user"},
                            "html_url": "https://github.com/user/repo",
                        }
                    ]
                },
            )
        return httpx.Response(429, json={"message": "Too Many Requests"})

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    with pytest.raises(ResearchError, match="rate limit exceeded"):
        await connector.search_repositories("query")


@pytest.mark.asyncio
async def test_github_network_error():
    """Network connection errors map to ResearchError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused by peer")

    transport = httpx.MockTransport(handler)
    connector = GitHubConnector(transport=transport)

    with pytest.raises(ResearchError, match="network error"):
        await connector.search_repositories("query")


@pytest.mark.asyncio
async def test_github_token_header_sent_only_when_set():
    """Authorization header is sent only when token is set and omitted otherwise."""
    recorded_headers: list[httpx.Headers] = []

    def handler(request: httpx.Request) -> httpx.Response:
        recorded_headers.append(request.headers)
        if request.url.path == "/search/repositories":
            return httpx.Response(200, json={"items": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    # 1. With token set
    connector_with_token = GitHubConnector(token="test-secret-gh-token", transport=transport)
    await connector_with_token.search_repositories("query")
    assert len(recorded_headers) == 1
    assert recorded_headers[0]["authorization"] == "Bearer test-secret-gh-token"

    # 2. Without token
    recorded_headers.clear()
    connector_no_token = GitHubConnector(token=None, transport=transport)
    await connector_no_token.search_repositories("query")
    assert len(recorded_headers) == 1
    assert "authorization" not in recorded_headers[0]
