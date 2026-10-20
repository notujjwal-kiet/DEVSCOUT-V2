"""
tests/test_source_relevance.py

Test Suite 1: Source Relevance Filtering
- Goal: "FastAPI Todo API"
- Candidates:
  * Ransomware repo -> rejected before Stage 5
  * Unrelated Codex config repo -> rejected before Stage 5
  * Official FastAPI docs -> accepted into reviewed resources
Zero network calls: local pure models and stage 4 runner.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
import pytest

from app.models.pipeline import (
    LearningGoal,
    Resource,
    ResourceType,
    SourceDiscoveryResult,
)
from app.pipeline import stage4_evidence_validation
from app.research.scoring import is_source_relevant_to_goal


def _make_resource(
    resource_id: str,
    url: str,
    title: str,
    rtype: ResourceType,
    summary: str = "",
    excerpt: str = "",
) -> Resource:
    return Resource(
        resource_id=resource_id,
        url=url,
        type=rtype,
        title=title,
        summary=summary,
        excerpt=excerpt,
        content_hash=f"hash_{resource_id}",
        reliability=0.8,
        freshness_score=0.9,
        retrieved_at=datetime.now(UTC),
        last_updated=datetime.now(UTC) - timedelta(days=20),
    )


def test_is_source_relevant_to_goal_filtering():
    """Unit checks on is_source_relevant_to_goal for malware and unrelated repos."""
    goal = LearningGoal(
        title="FastAPI Todo API",
        description="Build a RESTful todo API with FastAPI and Python",
        target_skills=["FastAPI", "REST API", "Pydantic"],
    )

    ransomware_repo = _make_resource(
        resource_id="res-ransomware",
        url="https://github.com/blackhat/ransomware-decryptor-payload",
        title="Automated Ransomware Encryption Payload",
        rtype=ResourceType.REPO,
        excerpt="Encrypts files and demands bitcoin ransom.",
    )
    codex_repo = _make_resource(
        resource_id="res-codex",
        url="https://github.com/developer/codex-config-dotfiles",
        title="Codex Config and Personal Dotfiles",
        rtype=ResourceType.REPO,
        excerpt="Zshrc, vimrc, and personal settings for codex environment.",
    )
    fastapi_docs = _make_resource(
        resource_id="res-fastapi",
        url="https://fastapi.tiangolo.com/tutorial/first-steps/",
        title="FastAPI Official Documentation - First Steps",
        rtype=ResourceType.DOCS,
        excerpt="FastAPI is a modern, fast web framework for building APIs with Python.",
    )

    rel_ransom, reason_ransom = is_source_relevant_to_goal(ransomware_repo, goal=goal)
    assert rel_ransom is False
    assert "malware" in reason_ransom.lower() or "threat" in reason_ransom.lower()

    rel_codex, reason_codex = is_source_relevant_to_goal(codex_repo, goal=goal)
    assert rel_codex is False
    assert "unrelated" in reason_codex.lower() or "zero conceptual overlap" in reason_codex.lower()

    rel_fastapi, _ = is_source_relevant_to_goal(fastapi_docs, goal=goal)
    assert rel_fastapi is True


@pytest.mark.asyncio
async def test_stage4_rejects_unrelated_and_threat_candidates():
    """Stage 4 routes ransomware and codex config repos to rejected_resources before stage 5."""
    goal = LearningGoal(
        title="FastAPI Todo API",
        description="Build a modern CRUD todo API using FastAPI and Pydantic",
        target_skills=["FastAPI", "Python", "CRUD API"],
    )

    ransomware_repo = _make_resource(
        resource_id="res-ransomware",
        url="https://github.com/threat-actor/python-ransomware-suite",
        title="Python Ransomware Suite",
        rtype=ResourceType.REPO,
        summary="A ransomware tool encrypting disk partitions",
    )
    codex_repo = _make_resource(
        resource_id="res-codex",
        url="https://github.com/myuser/codex-config-dotfiles",
        title="Codex Config and Terminal Dotfiles",
        rtype=ResourceType.REPO,
        summary="Configurations for bash, codex and zsh",
    )
    fastapi_docs = _make_resource(
        resource_id="res-fastapi",
        url="https://fastapi.tiangolo.com/tutorial/",
        title="FastAPI Official Tutorial",
        rtype=ResourceType.DOCS,
        summary="Official guide to building REST APIs with FastAPI",
    )

    discovery = SourceDiscoveryResult(
        goal_id=goal.goal_id,
        resources=[ransomware_repo, codex_repo, fastapi_docs],
        loop_count=0,
    )

    now = datetime.now(UTC)
    result = await stage4_evidence_validation.run(
        discovery,
        goal=goal,
        now=now,
    )

    rejected_ids = [r.resource_id for r in result.rejected_resources]
    reviewed_ids = [r.resource_id for r in result.reviewed_resources]

    assert "res-ransomware" in rejected_ids
    assert "res-codex" in rejected_ids
    assert "res-ransomware" not in reviewed_ids
    assert "res-codex" not in reviewed_ids

    assert "res-fastapi" in reviewed_ids
    assert "res-fastapi" not in rejected_ids

    # Confirmed that rejected resources are marked with is_rejected and rejection_reason
    rejected_ransom = next(r for r in result.rejected_resources if r.resource_id == "res-ransomware")
    assert rejected_ransom.is_rejected is True
    assert rejected_ransom.rejection_reason is not None

    rejected_codex = next(r for r in result.rejected_resources if r.resource_id == "res-codex")
    assert rejected_codex.is_rejected is True
    assert rejected_codex.rejection_reason is not None
