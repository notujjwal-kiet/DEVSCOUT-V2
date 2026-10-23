"""
app/providers/openai_compatible.py

OpenAI-compatible HTTP provider using httpx.AsyncClient to call
configurable /chat/completions endpoints (Groq, Cerebras, Ollama, etc.).

API keys and secrets are never logged or exposed.
Includes active verification and detailed diagnostic reporting (configuration,
initialization, authentication, model access, and quota).
"""
from __future__ import annotations

from typing import Any
import httpx
import structlog

from app.providers.base import (
    AuthenticationError,
    BaseChatProvider,
    ChatMessage,
    ChatResponse,
    ConfigurationError,
    InvalidRequestError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)

logger = structlog.get_logger(__name__)


def _parse_retry_after(header_val: str | None) -> float | None:
    """Parse Retry-After header value into seconds."""
    if not header_val:
        return None
    header_clean = header_val.strip()
    try:
        val = float(header_clean)
        return max(0.0, val)
    except ValueError:
        pass
    return 30.0


class OpenAICompatibleProvider(BaseChatProvider):
    """
    Generic provider for services implementing the OpenAI Chat Completions API specification:
    - Groq (https://api.groq.com/openai/v1)
    - Cerebras (https://api.cerebras.ai/v1)
    - Ollama (http://localhost:11434/v1)
    """

    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        tier: str = "primary",
        is_local: bool = False,
        price_in_per_1m: float | None = None,
        price_out_per_1m: float | None = None,
    ) -> None:
        super().__init__(
            name=name,
            model=model,
            tier=tier,
            is_local=is_local,
            timeout_seconds=timeout_seconds,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.price_in_per_1m = price_in_per_1m
        self.price_out_per_1m = price_out_per_1m
        # Diagnostic health state: "unconfigured", "ready", "authentication_failure",
        # "model_access_failure", "quota_exceeded", "provider_error"
        self.diagnostic_status: str = "ready" if (self.is_local or (self.api_key and self.base_url)) else "configuration_missing"
        self.diagnostic_detail: str = ""

    def is_available(self) -> bool:
        """Check if provider is configured and not permanently failed (auth or model access)."""
        if self.is_local:
            return bool(self.base_url)
        if not (self.api_key and self.base_url):
            return False
        # If verification or runtime failed with permanent auth or model access failure, flag unavailable
        if self.diagnostic_status in ("authentication_failure", "model_access_failure"):
            return False
        return True

    def get_info(self) -> dict[str, Any]:
        """Return safe public metadata with diagnostic information without credentials."""
        info: dict[str, Any] = {
            "id": f"{self.name}-{self.model}",
            "label": f"{self.name.capitalize()} ({self.model})",
            "provider": self.name,
            "local": self.is_local,
            "tier": self.tier,
            "available": self.is_available(),
            "used_today": 0,
            "limit_today": None,
            "price_in_per_1m": self.price_in_per_1m,
            "price_out_per_1m": self.price_out_per_1m,
        }
        if self.diagnostic_status:
            info["status"] = self.diagnostic_status
        if self.diagnostic_detail:
            info["status_detail"] = self.diagnostic_detail
        return info

    def _get_endpoint(self) -> str:
        """Compute the full chat completions endpoint URL."""
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    async def verify_availability(self) -> tuple[bool, str]:
        """
        Actively verify provider availability via real API check without exposing secrets.
        Distinguishes:
        - configuration: Missing API credentials or base URL
        - authentication: HTTP 401/403
        - model_access: HTTP 404 / model not found on account
        - quota: HTTP 429
        - ready: HTTP 200 with model verified
        """
        if not self.base_url:
            self.diagnostic_status = "configuration_missing"
            self.diagnostic_detail = "Missing base URL"
            return False, "configuration"

        if not self.is_local and not self.api_key:
            self.diagnostic_status = "configuration_missing"
            self.diagnostic_detail = "Missing API key in environment / .env"
            return False, "configuration"

        endpoint = f"{self.base_url}/models"
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_seconds, 10.0)) as client:
                resp = await client.get(endpoint, headers=headers)
        except httpx.TimeoutException:
            self.diagnostic_status = "timeout"
            self.diagnostic_detail = f"Verification timed out connecting to {self.name}"
            return False, "timeout"
        except httpx.RequestError as exc:
            self.diagnostic_status = "provider_error"
            self.diagnostic_detail = f"Network connection failed: {exc}"
            return False, "provider_error"

        if resp.status_code in (401, 403):
            self.diagnostic_status = "authentication_failure"
            self.diagnostic_detail = f"Authentication failed (HTTP {resp.status_code}). Invalid API credentials."
            logger.error("openai_compatible_verify_auth_failure", provider=self.name, status=resp.status_code)
            return False, "authentication"

        if resp.status_code == 429:
            self.diagnostic_status = "quota_exceeded"
            self.diagnostic_detail = "Rate limit or quota exhausted (HTTP 429)"
            return False, "quota"

        if resp.status_code == 200:
            try:
                models_data = resp.json().get("data", [])
                available_ids = [m.get("id") for m in models_data if isinstance(m, dict) and "id" in m]
                if self.model and available_ids and self.model not in available_ids:
                    self.diagnostic_status = "model_access_failure"
                    top_matches = [m for m in available_ids if "llama" in m or "gpt" in m or "qwen" in m][:3]
                    self.diagnostic_detail = (
                        f"Model '{self.model}' is not available on this account. "
                        f"Available options include: {', '.join(top_matches) if top_matches else ', '.join(available_ids[:3])}"
                    )
                    logger.warning(
                        "openai_compatible_model_not_found_on_account",
                        provider=self.name,
                        configured_model=self.model,
                        available=available_ids[:5],
                    )
                    return False, "model_access"
            except Exception:
                pass

            self.diagnostic_status = "ready"
            self.diagnostic_detail = f"Verified: model '{self.model}' is accessible"
            return True, "ready"

        self.diagnostic_status = "provider_error"
        self.diagnostic_detail = f"Unexpected response from {self.name} (HTTP {resp.status_code})"
        return False, "provider_error"

    async def chat(
        self,
        messages: list[ChatMessage],
        task: str = "chat",
        **kwargs: Any,
    ) -> ChatResponse:
        """Execute chat completion via HTTP POST with bounded timeout and safe error handling."""
        if not self.is_available():
            raise ConfigurationError(
                f"Provider '{self.name}' is unavailable ({self.diagnostic_detail or self.diagnostic_status}).",
                provider=self.name,
                model=self.model,
            )

        endpoint = self._get_endpoint()
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": kwargs.get("temperature", self.temperature),
        }
        if self.max_tokens or "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs.get("max_tokens", self.max_tokens)

        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]

        logger.debug(
            "openai_compatible_chat_request",
            provider=self.name,
            model=self.model,
            endpoint=endpoint,
            message_count=len(messages),
        )

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(endpoint, headers=headers, json=payload)
        except httpx.TimeoutException as exc:
            logger.warning(
                "openai_compatible_timeout",
                provider=self.name,
                model=self.model,
                timeout=self.timeout_seconds,
            )
            raise ProviderTimeoutError(
                f"Request to {self.name} timed out after {self.timeout_seconds}s.",
                provider=self.name,
                model=self.model,
            ) from exc
        except httpx.RequestError as exc:
            logger.warning(
                "openai_compatible_network_error",
                provider=self.name,
                model=self.model,
                error=str(exc),
            )
            raise ProviderUnavailableError(
                f"Network connection to {self.name} failed: {exc}",
                provider=self.name,
                model=self.model,
            ) from exc

        status_code = response.status_code

        # ── Error Mapping ─────────────────────────────────────────────
        if status_code in (401, 403):
            self.diagnostic_status = "authentication_failure"
            self.diagnostic_detail = f"Authentication failed (HTTP {status_code})"
            logger.error(
                "openai_compatible_auth_error",
                provider=self.name,
                model=self.model,
                status=status_code,
            )
            raise AuthenticationError(
                f"Authentication failed for {self.name} (HTTP {status_code}). Check API credentials.",
                provider=self.name,
                model=self.model,
            )

        if status_code == 404:
            err_msg = response.text[:200]
            self.diagnostic_status = "model_access_failure"
            self.diagnostic_detail = f"Model '{self.model}' not found on account (HTTP 404)"
            logger.error(
                "openai_compatible_model_access_error",
                provider=self.name,
                model=self.model,
                status=status_code,
                detail=err_msg,
            )
            raise InvalidRequestError(
                f"Model '{self.model}' does not exist or account lacks access on {self.name} (HTTP 404).",
                provider=self.name,
                model=self.model,
            )

        if status_code == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            self.diagnostic_status = "quota_exceeded"
            self.diagnostic_detail = f"Rate limit or quota exhausted (HTTP 429)"
            logger.warning(
                "openai_compatible_rate_limit",
                provider=self.name,
                model=self.model,
                retry_after=retry_after,
            )
            raise RateLimitError(
                f"Rate limit exceeded for {self.name} (HTTP 429).",
                provider=self.name,
                model=self.model,
                retry_after=retry_after,
            )

        if status_code in (400, 422):
            err_msg = response.text[:200]
            logger.warning(
                "openai_compatible_invalid_request",
                provider=self.name,
                model=self.model,
                status=status_code,
                detail=err_msg,
            )
            raise InvalidRequestError(
                f"Invalid request to {self.name} (HTTP {status_code}): {err_msg}",
                provider=self.name,
                model=self.model,
            )

        if status_code >= 500:
            logger.warning(
                "openai_compatible_server_error",
                provider=self.name,
                model=self.model,
                status=status_code,
            )
            raise ProviderUnavailableError(
                f"{self.name} service is temporarily unavailable (HTTP {status_code}).",
                provider=self.name,
                model=self.model,
            )

        if status_code != 200:
            raise ProviderUnavailableError(
                f"Unexpected status from {self.name} (HTTP {status_code}).",
                provider=self.name,
                model=self.model,
            )

        # ── Success Parsing ───────────────────────────────────────────
        try:
            data = response.json()
            choice = data["choices"][0]
            message_obj = choice.get("message", {})
            content = message_obj.get("content", "")
            finish_reason = choice.get("finish_reason")
            usage = data.get("usage")
            self.diagnostic_status = "ready"
            self.diagnostic_detail = "Operational"
        except (KeyError, IndexError, ValueError) as exc:
            logger.error(
                "openai_compatible_malformed_response",
                provider=self.name,
                model=self.model,
                error=str(exc),
            )
            raise ProviderUnavailableError(
                f"Failed to parse response from {self.name}.",
                provider=self.name,
                model=self.model,
            ) from exc

        return ChatResponse(
            content=content,
            provider=self.name,
            model=self.model,
            task=task,
            research_backed=False,
            finish_reason=finish_reason,
            usage=usage,
        )
