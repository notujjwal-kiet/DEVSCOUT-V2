"""
app/pipeline/structured_llm.py

Utility for invoking GeminiProvider.generate_structured with schema validation,
single-retry error reflection, and typed error handling.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypeVar
from pydantic import BaseModel, ValidationError
import structlog

from app.llm.provider import LLMProvider, LLMProviderError, RateLimitError

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(LLMProviderError):
    """Raised when structured output generation fails schema validation after retry."""


def is_parse_or_validation_error(exc: Exception) -> bool:
    """Return True only if exc is a parse or schema/output validation error."""
    if isinstance(exc, RateLimitError):
        return False
    if isinstance(exc, LLMProviderError):
        msg = str(exc).lower()
        return "json" in msg or "parse" in msg or "validation" in msg
    if isinstance(exc, (ValidationError, json.JSONDecodeError, ValueError)):
        return True
    return False


async def generate_structured_with_retry(
    provider: LLMProvider | object,
    prompt: str,
    schema: type[T],
    system_prompt: str | None = None,
    validator: Callable[[T], tuple[bool, str]] | None = None,
    **kwargs: object,
) -> T:
    """
    Generate structured output adhering to schema using GeminiProvider.
    Retries once ONLY on parse or validation errors with the error appended to the prompt.
    Non-parse/validation errors (e.g. RateLimitError, network, auth) are re-raised immediately without retrying.
    If the retry fails, raises StructuredOutputError.
    """
    try:
        result = await provider.generate_structured(
            prompt=prompt,
            schema=schema,
            system_prompt=system_prompt,
            **kwargs,
        )
        if validator is not None:
            is_valid, err_msg = validator(result)
            if not is_valid:
                raise ValueError(f"Output validation failed: {err_msg}")
        return result
    except Exception as first_exc:
        if not is_parse_or_validation_error(first_exc):
            raise first_exc

        logger.warning(
            "structured_output_first_attempt_failed",
            schema=schema.__name__,
            error=str(first_exc),
        )
        retry_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Your previous output failed validation with error:\n"
            f"{str(first_exc)}\n"
            f"Please correct the error and strictly adhere to the requested schema."
        )
        try:
            retry_result = await provider.generate_structured(
                prompt=retry_prompt,
                schema=schema,
                system_prompt=system_prompt,
                **kwargs,
            )
            if validator is not None:
                is_valid, err_msg = validator(retry_result)
                if not is_valid:
                    raise ValueError(f"Output validation failed on retry: {err_msg}")
            return retry_result
        except Exception as retry_exc:
            if isinstance(retry_exc, RateLimitError):
                raise retry_exc

            logger.error(
                "structured_output_retry_failed",
                schema=schema.__name__,
                error=str(retry_exc),
            )
            raise StructuredOutputError(
                f"Failed to generate valid structured output for {schema.__name__} after retry: {retry_exc}"
            ) from retry_exc

