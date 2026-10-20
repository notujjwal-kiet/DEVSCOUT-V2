"""
tests/test_evidence_gate_e2e.py

End-to-end orchestrator tests with fake connectors:
1. A FastAPI Todo goal where discovery returns:
   - a ransomware repo
   - a Codex-config repo
   - real FastAPI docs
   -> Only the docs reach Stage 5 and LearningResponse.resources.
2. All-irrelevant candidates (only ransomware + codex config):
   -> evidence_status is "insufficient", confidence_score <= 0.35,
      critical warning NO_RELIABLE_SOURCES is present,
      and uncited claims are marked unsupported.
Zero real network calls: mocked discovery, web fetch, and local fake LLM provider.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
import pytest

from app.models.pipeline import (
    ExperienceLevel,
    LearnRequest,
    LearningResponse,
    ResourceType,
)
from app.pipeline import orchestrator
from app.research.models import DiscoveredSource, ResearchDocument
from app.models.pipeline import Resource


class _FakeE2EProvider:
    """Fake Gemini provider returning valid structured JSON payloads for pipeline stages."""

    async def generate_structured(
        self,
        prompt: str,
        schema: type | None = None,
        response_schema: type | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        target_schema = schema or response_schema
        name = getattr(target_schema, "__name__", "")
        if "LearningPlan" in name:
            return target_schema(
                goal_id="goal-fastapi-todo",
                knowledge_gaps=["Routing", "Validation"],
                learning_outcomes=["Build FastAPI Todo API"],
                depth_level="intermediate",
                milestones=[
                    {
                        "milestone_id": "m1",
                        "title": "FastAPI Basics",
                        "description": "Learn routes and validation",
                        "target_outcome": "Working endpoints",
                        "concepts": ["FastAPI", "Routing"],
                    }
                ],
            )
        elif "ExplanationResult" in name or "Explanation" in name:
            return response_schema(
                topic="FastAPI Todo API",
                summary="Guide to building a RESTful Todo API using FastAPI.",
                sections=[
                    {
                        "title": "CRUD Endpoints",
                        "content": "FastAPI provides declarative route definitions.",
                        "citations": [],
                    }
                ],
                confidence_score=0.95,
            )
        elif "UnderstandingChecklist" in name or "Question" in name:
            return response_schema(
                questions=[
                    {
                        "question_id": "q1",
                        "concept_id": "c1",
                        "question_text": "How does FastAPI handle validation?",
                        "question_type": "conceptual",
                        "expected_points": ["Pydantic models", "Type annotations"],
                    },
                    {
                        "question_id": "q2",
                        "concept_id": "c2",
                        "question_text": "What server runs FastAPI applications?",
                        "question_type": "conceptual",
                        "expected_points": ["Uvicorn", "ASGI"],
                    },
                    {
                        "question_id": "q3",
                        "concept_id": "c3",
                        "question_text": "How do you declare route path parameters?",
                        "question_type": "code_analysis",
                        "expected_points": ["Path parameters", "Type hints"],
                    },
                ]
            )
        # Default fallback
        try:
            return response_schema()
        except Exception:
            return None


@pytest.mark.asyncio
async def test_e2e_fastapi_todo_only_real_docs_reach_stage5(monkeypatch):
    """
    FastAPI Todo goal with discovery returning ransomware repo, codex-config repo,
    and real FastAPI docs. Only the docs survive the gate to reach Stage 5.
    """
    fake_provider = _FakeE2EProvider()

    # Mock discovery to return ransomware repo, codex config repo, and real FastAPI docs
    async def fake_discover(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://github.com/blackhat/ransomware-tool",
                title="Automated Ransomware Decryptor Exploit",
                domain="github.com",
                source_type_hint=ResourceType.REPO,
            ),
            DiscoveredSource(
                url="https://github.com/developer/codex-config-dotfiles",
                title="Codex Config and Personal Dotfiles",
                domain="github.com",
                source_type_hint=ResourceType.REPO,
            ),
            DiscoveredSource(
                url="https://fastapi.tiangolo.com/tutorial/todos/",
                title="FastAPI Official Documentation - Todos",
                domain="fastapi.tiangolo.com",
                source_type_hint=ResourceType.DOCS,
            ),
        ]

    async def fake_fetch(url, resource_type, client=None):
        if "ransomware" in url:
            res = Resource(
                url=url,
                type=ResourceType.REPO,
                title="Automated Ransomware Decryptor Exploit",
                excerpt="Encrypts disk files and demands payment in cryptocurrency.",
                content_hash="hash_rw",
                last_updated=datetime.now(UTC),
                reliability=0.5,
                freshness_score=0.9,
                topic="FastAPI Basics",
            )
            return ResearchDocument(resource=res, text="Malicious ransomware payload text.")
        elif "codex-config" in url:
            res = Resource(
                url=url,
                type=ResourceType.REPO,
                title="Codex Config and Personal Dotfiles",
                excerpt="Configuration settings, zshrc, and vimrc dotfiles for codex agent.",
                content_hash="hash_codex",
                last_updated=datetime.now(UTC),
                reliability=0.6,
                freshness_score=0.9,
                topic="FastAPI Basics",
            )
            return ResearchDocument(resource=res, text="Zshrc, vimrc, personal dotfiles.")
        else:
            res = Resource(
                url=url,
                type=ResourceType.DOCS,
                title="FastAPI Official Documentation - Todos",
                excerpt="FastAPI is a modern web framework for building APIs with Python.",
                content_hash="hash_fastapi",
                last_updated=datetime.now(UTC),
                reliability=1.0,
                freshness_score=0.95,
                topic="FastAPI Basics",
            )
            return ResearchDocument(resource=res, text="FastAPI tutorial building a Todo REST API with Python.")

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", fake_discover)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", fake_fetch)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", AsyncMock(return_value=[]))

    request = LearnRequest(
        goal="Build a FastAPI Todo API",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["FastAPI", "Python"],
    )

    response = await orchestrator.run_learning_pipeline(request, provider=fake_provider)

    assert isinstance(response, LearningResponse)
    # Only the real FastAPI docs survive the gate
    surviving_urls = [r.url for r in response.resources]
    assert "https://fastapi.tiangolo.com/tutorial/todos/" in surviving_urls
    assert "https://github.com/blackhat/ransomware-tool" not in surviving_urls
    assert "https://github.com/developer/codex-config-dotfiles" not in surviving_urls
    assert len(response.resources) == 1


@pytest.mark.asyncio
async def test_e2e_all_irrelevant_candidates_evidence_insufficient_and_critical_warning(monkeypatch):
    """
    All candidate sources are irrelevant or malicious.
    Verifies:
    - evidence_status == "insufficient"
    - confidence_score <= 0.35
    - NO_RELIABLE_SOURCES critical warning is present
    - Uncited explanation claims are marked unsupported
    """
    fake_provider = _FakeE2EProvider()

    async def fake_discover(query, max_results=5, provider=None, http_client=None):
        return [
            DiscoveredSource(
                url="https://github.com/blackhat/ransomware-tool",
                title="Automated Ransomware Decryptor Exploit",
                domain="github.com",
                source_type_hint=ResourceType.REPO,
            ),
            DiscoveredSource(
                url="https://github.com/developer/codex-config-dotfiles",
                title="Codex Config and Personal Dotfiles",
                domain="github.com",
                source_type_hint=ResourceType.REPO,
            ),
        ]

    async def fake_fetch(url, resource_type, client=None):
        if "ransomware" in url:
            res = Resource(
                url=url,
                type=ResourceType.REPO,
                title="Automated Ransomware Decryptor Exploit",
                excerpt="Encrypts disk files and demands payment in cryptocurrency.",
                content_hash="hash_rw",
                last_updated=datetime.now(UTC),
                reliability=0.5,
                freshness_score=0.9,
                topic="FastAPI Basics",
            )
            return ResearchDocument(resource=res, text="Malicious ransomware payload text.")
        else:
            res = Resource(
                url=url,
                type=ResourceType.REPO,
                title="Codex Config and Personal Dotfiles",
                excerpt="Configuration settings, zshrc, and vimrc dotfiles for codex agent.",
                content_hash="hash_codex",
                last_updated=datetime.now(UTC),
                reliability=0.6,
                freshness_score=0.9,
                topic="FastAPI Basics",
            )
            return ResearchDocument(resource=res, text="Zshrc, vimrc, personal dotfiles.")

    monkeypatch.setattr("app.pipeline.stage3_research.discover_sources", fake_discover)
    monkeypatch.setattr("app.pipeline.stage3_research.fetch_page", fake_fetch)
    monkeypatch.setattr("app.research.github.GitHubConnector.search_repositories", AsyncMock(return_value=[]))

    request = LearnRequest(
        goal="Build a FastAPI Todo API",
        experience_level=ExperienceLevel.INTERMEDIATE,
        tech_stack=["FastAPI", "Python"],
    )

    response = await orchestrator.run_learning_pipeline(request, provider=fake_provider)

    assert isinstance(response, LearningResponse)
    # Zero sources survive
    assert len(response.resources) == 0

    # Evidence shortage assertions
    assert response.evidence_status == "insufficient"
    assert response.confidence_score <= 0.35

    warning_codes = {w.code: w.severity for w in response.warnings}
    assert "NO_RELIABLE_SOURCES" in warning_codes
    assert warning_codes["NO_RELIABLE_SOURCES"] == "critical"

    # Uncited claims are not presented as grounded
    assert response.explanation is not None
    for section in response.explanation.sections:
        if not section.citations:
            assert section.support_status == "unsupported"
