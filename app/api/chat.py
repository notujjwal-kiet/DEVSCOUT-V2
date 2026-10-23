"""
app/api/chat.py

API routes for multi-provider AI chat:
- POST /api/chat: Multi-provider chat with task routing and automatic fallback.
"""
from __future__ import annotations

from typing import Literal, Self
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
import structlog

from app.api.auth import verify_api_key_and_rate_limit
from app.providers.base import (
    ChatMessage,
    ChatResponse,
    ProviderError,
    ProviderUnavailableError,
)
from app.providers.router import get_provider_router

logger = structlog.get_logger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key_and_rate_limit)])


class ChatMessageInput(BaseModel):
    """Input message for chat."""

    role: Literal["system", "user", "assistant"] = Field(
        default="user", description="Author role"
    )
    content: str = Field(
        ..., min_length=1, max_length=100000, description="Message text"
    )


class ChatRequest(BaseModel):
    """Request payload for POST /api/chat."""

    message: str | None = Field(
        default=None, description="Direct user message prompt"
    )
    messages: list[ChatMessageInput] | None = Field(
        default=None, description="Optional full conversation history"
    )
    task: Literal["chat", "code", "explain", "research"] = Field(
        default="chat", description="Task type for intelligent routing"
    )
    model: str | None = Field(
        default=None, description="Optional preferred model or provider ID"
    )

    @model_validator(mode="after")
    def validate_message_or_messages(self) -> Self:
        if not self.message and not self.messages:
            raise ValueError("Either 'message' or 'messages' must be provided.")
        return self


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Multi-provider AI chat completion with automatic fallback",
)
async def chat_completion(request: ChatRequest) -> ChatResponse:
    """
    Execute chat completion across Groq, Cerebras, Gemini, and Ollama.
    Automatically handles task routing and fallback on rate limits or outages.
    """
    router_instance = get_provider_router()

    chat_messages: list[ChatMessage] = []
    if request.messages:
        for m in request.messages:
            chat_messages.append(ChatMessage(role=m.role, content=m.content))
    if request.message:
        chat_messages.append(ChatMessage(role="user", content=request.message))

    try:
        response = await router_instance.route_chat(
            messages=chat_messages,
            task=request.task,
            preferred_model_or_provider=request.model,
        )
        return response
    except ProviderUnavailableError as exc:
        logger.error("chat_all_providers_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"AI service temporarily unavailable: {exc.message}",
        )
    except ProviderError as exc:
        logger.error("chat_provider_error", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Provider error: {exc.message}",
        )
    except Exception as exc:
        logger.error("chat_unexpected_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing the chat request.",
        )
