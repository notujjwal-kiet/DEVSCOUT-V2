"""
app/research/github.py

Async GitHubConnector using httpx against the GitHub REST API.
Discovers repositories by search query, fetches and decodes README content,
and builds ResearchDocument instances for stage 4 evidence review.
"""
from __future__ import annotations

import base64
from datetime import UTC, datetime
import hashlib
import os

import httpx

from app.models.pipeline import Resource, ResourceType
from app.research.models import ResearchDocument, ResearchError

MAX_README_CHARS = 20000


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing consecutive whitespace into single spaces."""
    return " ".join(text.split())


def hash_content(text: str) -> str:
    """Compute SHA-256 digest of whitespace-normalized text."""
    return hashlib.sha256(normalize_whitespace(text).encode("utf-8")).hexdigest()


class GitHubConnector:
    """
    Async connector to GitHub REST API.
    Fetches repositories and READMEs to build ResearchDocuments.
    """

    def __init__(
        self,
        token: str | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        self.token = (
            token
            or os.getenv("GITHUB_TOKEN")
            or (getattr(cfg, "github_token", None) if cfg else None)
        )
        if client is not None:
            self._client = client
        elif transport is not None:
            self._client = httpx.AsyncClient(transport=transport)
        else:
            self._client = None

    def _get_headers(self) -> dict[str, str]:
        """Construct request headers. Authorization header sent only when token is set."""
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "DEVSCOUT-Learning-Companion",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def search_repositories(
        self,
        query: str,
        limit: int = 5,
    ) -> list[ResearchDocument]:
        """
        Search GitHub repositories sorted by stars, fetch READMEs, and build ResearchDocuments.
        Skips repositories that do not have a README.
        Raises ResearchError on rate limit (403/429) or network errors.
        """
        headers = self._get_headers()
        search_url = "https://api.github.com/search/repositories"
        params: dict[str, object] = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": limit,
        }

        try:
            if self._client is not None:
                response = await self._client.get(search_url, params=params, headers=headers)
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.get(search_url, params=params, headers=headers)
        except httpx.RequestError as exc:
            raise ResearchError(f"GitHub API network error searching repositories: {exc}") from exc

        if response.status_code in (403, 429):
            raise ResearchError(
                f"GitHub API rate limit exceeded ({response.status_code}): {response.text}"
            )
        if response.status_code >= 400:
            raise ResearchError(
                f"GitHub API error searching repositories ({response.status_code}): {response.text}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise ResearchError(f"GitHub API invalid JSON response: {exc}") from exc

        items = data.get("items", [])
        documents: list[ResearchDocument] = []

        for item in items:
            owner_data = item.get("owner", {})
            owner = owner_data.get("login") if isinstance(owner_data, dict) else ""
            repo_name = item.get("name", "")
            full_name = item.get("full_name") or (f"{owner}/{repo_name}" if owner and repo_name else repo_name)
            html_url = item.get("html_url") or f"https://github.com/{full_name}"
            title = full_name
            popularity = item.get("stargazers_count", 0)

            pushed_at_str = item.get("pushed_at")
            last_updated: datetime | None = None
            if pushed_at_str:
                try:
                    last_updated = datetime.fromisoformat(pushed_at_str.replace("Z", "+00:00"))
                except Exception:
                    last_updated = None

            readme_url = f"https://api.github.com/repos/{owner}/{repo_name}/readme"
            try:
                if self._client is not None:
                    readme_resp = await self._client.get(readme_url, headers=headers)
                else:
                    async with httpx.AsyncClient() as client:
                        readme_resp = await client.get(readme_url, headers=headers)
            except httpx.RequestError as exc:
                raise ResearchError(
                    f"GitHub API network error fetching README for '{full_name}': {exc}"
                ) from exc

            # Skip repos with no README instead of failing
            if readme_resp.status_code == 404:
                continue

            if readme_resp.status_code in (403, 429):
                raise ResearchError(
                    f"GitHub API rate limit exceeded ({readme_resp.status_code}): {readme_resp.text}"
                )

            if readme_resp.status_code >= 400:
                raise ResearchError(
                    f"GitHub API error fetching README for '{full_name}' ({readme_resp.status_code}): {readme_resp.text}"
                )

            try:
                readme_json = readme_resp.json()
            except Exception as exc:
                raise ResearchError(
                    f"Failed to parse README response JSON for '{full_name}': {exc}"
                ) from exc

            content_b64 = readme_json.get("content", "")
            encoding = readme_json.get("encoding", "base64")
            if encoding == "base64" and content_b64:
                try:
                    decoded_bytes = base64.b64decode(content_b64)
                    raw_text = decoded_bytes.decode("utf-8", errors="replace")
                except Exception:
                    raw_text = content_b64
            else:
                raw_text = content_b64

            # Truncate text to 20,000 chars
            truncated_text = raw_text[:MAX_README_CHARS]
            content_hash = hash_content(truncated_text)
            retrieved_at = datetime.now(UTC)

            resource = Resource(
                url=html_url,
                type=ResourceType.REPO,
                title=title,
                content_hash=content_hash,
                retrieved_at=retrieved_at,
            )

            doc = ResearchDocument(
                resource=resource,
                text=truncated_text,
                last_updated=last_updated,
                popularity=popularity,
            )
            documents.append(doc)

        return documents
