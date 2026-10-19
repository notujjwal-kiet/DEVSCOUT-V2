"""
app/pipeline/session_store.py

Session persistence layer for DEVSCOUT Learning Companion.
Stores LearningResponse, Gemini-given snippets & hashes, understanding results,
and code attempt reviews keyed by response_id.
Defined as a clean interface so Postgres (or Redis) can replace it seamlessly.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from app.models.pipeline import (
    AttemptReview,
    LearningResponse,
    UnderstandingResult,
)
from app.policy.attempt_review import hash_code


def _utcnow() -> datetime:
    return datetime.now(UTC)


class SessionData(BaseModel):
    """
    Complete state of a learner's active session.
    Holds the LearningResponse, snippets & their SHA-256 hashes, understanding checks,
    and history of code attempt reviews.
    """

    response_id: str
    learning_response: LearningResponse
    snippet_hashes: list[str] = Field(default_factory=list)
    understanding_results: list[UnderstandingResult] = Field(default_factory=list)
    attempt_reviews: list[AttemptReview] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class SessionStore(ABC):
    """Abstract interface for session storage."""

    @abstractmethod
    async def get(self, response_id: str) -> SessionData | None:
        """Retrieve session data by response_id."""
        pass

    @abstractmethod
    async def save(self, session: SessionData) -> None:
        """Persist or update session data."""
        pass

    @abstractmethod
    async def delete(self, response_id: str) -> bool:
        """Delete session data. Return True if deleted, False if not found."""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all stored sessions (useful for tests)."""
        pass


class InMemorySessionStore(SessionStore):
    """In-memory implementation of SessionStore."""

    def __init__(self) -> None:
        self._store: dict[str, SessionData] = {}

    async def get(self, response_id: str) -> SessionData | None:
        return self._store.get(response_id)

    async def save(self, session: SessionData) -> None:
        session.updated_at = _utcnow()
        self._store[session.response_id] = session

    async def delete(self, response_id: str) -> bool:
        return self._store.pop(response_id, None) is not None

    async def clear(self) -> None:
        self._store.clear()


# Default singleton instance
_GLOBAL_SESSION_STORE: SessionStore = InMemorySessionStore()


def get_session_store() -> SessionStore:
    """Return the global SessionStore instance."""
    return _GLOBAL_SESSION_STORE


def create_session_from_response(
    response: LearningResponse,
    extra_hashes: list[str] | None = None,
) -> SessionData:
    """
    Create a SessionData object from a LearningResponse, computing SHA-256 hashes
    for all contained code snippets.
    """
    snippet_hashes: list[str] = list(extra_hashes or [])

    for snippet in response.snippets:
        h = hash_code(snippet.code)
        if h not in snippet_hashes:
            snippet_hashes.append(h)

    understanding_results: list[UnderstandingResult] = []
    if response.understanding_result:
        understanding_results.append(response.understanding_result)

    return SessionData(
        response_id=response.response_id,
        learning_response=response,
        snippet_hashes=snippet_hashes,
        understanding_results=understanding_results,
        attempt_reviews=[],
    )
