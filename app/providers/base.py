"""
app/providers/base.py

Common asynchronous chat-provider interface and response definitions.
No API keys or secrets are ever logged or exposed in exceptions.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Normalized chat message."""

    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="Role of the message author"
    )
    content: str = Field(..., description="Text content of the message")


class ChatResponse(BaseModel):
    """Normalized typed response from any chat provider."""

    content: str = Field(..., description="Generated text content")
    provider: str = Field(..., description="Provider identifier (e.g. groq, cerebras, gemini, ollama)")
    model: str = Field(..., description="Model name used")
    task: str = Field(default="chat", description="Task type (chat, code, explain, research)")
    research_backed: bool = Field(
        default=False,
        description="True only if verified research/grounding evidence was utilized",
    )
    finish_reason: str | None = Field(default=None, description="Reason generation finished")
    usage: dict[str, Any] | None = Field(
        default=None, description="Token usage statistics if provided by provider"
    )


# ── Exception Hierarchy ──────────────────────────────────────────────


class ProviderError(Exception):
    """Base exception for all AI provider operations."""

    def __init__(self, message: str, provider: str = "", model: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model


class AuthenticationError(ProviderError):
    """401/403: Invalid, expired, or missing credentials. Non-retryable on same provider."""

    pass


class InvalidRequestError(ProviderError):
    """400/422: Malformed payload, unsupported parameters, or context window exceeded."""

    pass


class RateLimitError(ProviderError):
    """429: Rate limit or daily quota exhausted. Eligible for fallback."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        model: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, model=model)
        self.retry_after = retry_after


class ProviderUnavailableError(ProviderError):
    """5xx or network connection failure. Eligible for fallback."""

    pass


class ProviderTimeoutError(ProviderError):
    """Request exceeded bounded timeout limit. Eligible for fallback."""

    pass


class ConfigurationError(ProviderError):
    """Provider is missing mandatory configuration or credentials."""

    pass


# ── Abstract Base Provider ───────────────────────────────────────────


class BaseChatProvider(ABC):
    """Abstract base class for all AI chat providers."""

    def __init__(
        self,
        name: str,
        model: str,
        tier: str = "primary",
        is_local: bool = False,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.name = name
        self.model = model
        self.tier = tier
        self.is_local = is_local
        self.timeout_seconds = timeout_seconds

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        task: str = "chat",
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute chat completion and return normalized ChatResponse."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if provider has valid configuration and is ready."""
        raise NotImplementedError

    @abstractmethod
    def get_info(self) -> dict[str, Any]:
        """Return public metadata dictionary for GET /api/models without secrets."""
        raise NotImplementedError
