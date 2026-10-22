"""
app/llm/provider.py

LLM Provider interface and stubs for DEVSCOUT.
Protocol LLMProvider with Ollama and Gemini stub implementations (no network calls).
Provider is chosen by the LLM_PROVIDER environment variable; model names are configurable.
"""
from __future__ import annotations

import asyncio
import os
import random
import re
from typing import Callable, Protocol, TypeVar, runtime_checkable

from google import genai
from google.genai import errors, types
import httpx
from pydantic import BaseModel
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMProviderError(ValueError):
    """Raised when an LLM provider encounters a network, connection, configuration, or execution error."""


class RateLimitError(LLMProviderError):
    """Raised when an LLM provider encounters a 429 rate limit or quota exhaustion error."""


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the LLM provider interface."""

    model_name: str

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs: object) -> str:
        """Generate plain text from prompt."""
        ...

    async def generate_structured(
        self, prompt: str, schema: type[T], system_prompt: str | None = None, **kwargs: object
    ) -> T:
        """Generate structured output adhering to a Pydantic schema."""
        ...


class OllamaProvider:
    """Ollama local model provider using httpx async HTTP API."""

    def __init__(
        self,
        host: str | None = None,
        model_name: str | None = None,
        think: bool | None = None,
        timeout: float | None = None,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        model_id: str = "ollama-local",
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        self.model_id = model_id

        self.host = (
            host
            or os.getenv("OLLAMA_HOST")
            or (cfg.ollama_host if cfg else "http://localhost:11434")
        )
        self.model_name = (
            model_name
            or os.getenv("OLLAMA_MODEL")
            or (cfg.ollama_model if cfg else "llama3.2")
        )
        if think is not None:
            self.think = think
        elif os.getenv("OLLAMA_THINK") is not None:
            self.think = os.getenv("OLLAMA_THINK", "false").lower() in ("true", "1", "yes")
        elif cfg is not None and hasattr(cfg, "ollama_think"):
            self.think = cfg.ollama_think
        else:
            self.think = False

        if timeout is not None:
            self.timeout = float(timeout)
        elif os.getenv("OLLAMA_TIMEOUT") is not None:
            self.timeout = float(os.getenv("OLLAMA_TIMEOUT"))
        elif cfg is not None and hasattr(cfg, "ollama_timeout"):
            self.timeout = float(cfg.ollama_timeout)
        else:
            self.timeout = 120.0

        if client is not None:
            self._client = client
        elif transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=self.timeout)
        else:
            self._client = None

        self.last_response: dict[str, object] | None = None
        self.last_usage_data: dict[str, object] | None = None

    async def _post_chat(self, payload: dict[str, object]) -> httpx.Response:
        url = f"{self.host.rstrip('/')}/api/chat"
        try:
            if self._client is not None:
                response = await self._client.post(url, json=payload)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(url, json=payload)
            response.raise_for_status()
            return response
        except httpx.RequestError as exc:
            raise LLMProviderError(
                f"Failed to connect to Ollama host at '{self.host}': {exc}"
            ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 or "quota" in exc.response.text.lower():
                raise RateLimitError(
                    f"Ollama host '{self.host}' rate limit exceeded (HTTP 429): {exc.response.text}"
                ) from exc
            raise LLMProviderError(
                f"Ollama host '{self.host}' returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> str:
        """Generate plain text from prompt using Ollama chat endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,
        }
        response = await self._post_chat(payload)
        try:
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to parse JSON response from Ollama host at '{self.host}': {exc}"
            ) from exc

        message = data.get("message")
        self.last_response = data
        self.last_usage_data = data
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(data.get("response", ""))

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> T:
        """Generate structured output adhering to schema using Ollama chat endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "format": schema.model_json_schema(),
        }
        response = await self._post_chat(payload)
        try:
            data = response.json()
        except Exception as exc:
            raise LLMProviderError(
                f"Failed to parse JSON response from Ollama host at '{self.host}': {exc}"
            ) from exc

        self.last_response = data
        self.last_usage_data = data
        message = data.get("message")
        if isinstance(message, dict):
            content = str(message.get("content", ""))
        else:
            content = str(data.get("response", ""))

        return schema.model_validate_json(content)


class OpenAICompatibleProvider:
    """
    OpenAI-compatible LLM provider using httpx async client.
    Interacts with POST {base_url}/chat/completions using standard messages and temperature 0.2.
    Supports JSON mode and structured output adhering to Pydantic schemas.
    """

    def __init__(
        self,
        base_url: str,
        api_key_env: str | None = None,
        model_name: str | None = None,
        model_id: str = "openai-compatible",
        supports_json_schema: bool = False,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        rate_gate: object | None = None,
    ):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key_env = api_key_env
        self.model_name = model_name or "default"
        self.model_id = model_id
        self.supports_json_schema = supports_json_schema
        self.timeout = float(timeout)
        self.rate_gate = rate_gate

        if client is not None:
            self._client = client
        elif transport is not None:
            self._client = httpx.AsyncClient(transport=transport, timeout=self.timeout)
        else:
            self._client = None

        self.last_response: dict[str, object] | None = None
        self.last_usage_data: dict[str, object] | None = None

    def _get_api_key(self) -> str | None:
        """Read API key from os.environ[api_key_env] at call time. Never log or include in errors."""
        if not self.api_key_env:
            return None
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise LLMProviderError(
                f"Authentication failed: environment variable '{self.api_key_env}' is not set."
            )
        return key

    async def _post_chat(self, payload: dict[str, object]) -> dict[str, object]:
        """POST {base_url}/chat/completions with rate gate acquire and error mapping."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        key = self._get_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"

        rate_gate = self.rate_gate
        if rate_gate is None:
            try:
                from app.llm.rate_gate import get_rate_gate

                rate_gate = get_rate_gate()
            except Exception:
                pass

        max_retries = 2  # max 2 retries for transient 5xx errors (3 attempts total)
        attempt = 0

        while True:
            attempt += 1
            # Count every attempt against the model's sliding window RPM and daily budget
            if rate_gate is not None and hasattr(rate_gate, "acquire"):
                await rate_gate.acquire(self.model_id)

            try:
                if self._client is not None:
                    response = await self._client.post(url, json=payload, headers=headers)
                else:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(url, json=payload, headers=headers)
            except httpx.RequestError as exc:
                logger.error("openai_compatible_network_error", model_id=self.model_id, error=str(exc))
                raise LLMProviderError(
                    f"Network error connecting to OpenAI-compatible endpoint at '{self.base_url}': {exc}"
                ) from exc

            status = response.status_code
            if 200 <= status < 300:
                try:
                    data = response.json()
                except Exception as exc:
                    raise LLMProviderError(
                        f"Failed to parse JSON response from '{self.base_url}': {exc}"
                    ) from exc

                self.last_response = data
                usage = data.get("usage")
                if isinstance(usage, dict):
                    self.last_usage_data = usage
                return data

            # 401 / 403 -> Authentication error (Never log or include key)
            if status in (401, 403):
                logger.warning(
                    "openai_compatible_auth_failed",
                    model_id=self.model_id,
                    status_code=status,
                )
                raise LLMProviderError(
                    f"Authentication failed (HTTP {status}) for model '{self.model_name}' on '{self.base_url}'."
                )

            # 429 -> RateLimitError (Not retried)
            if status == 429:
                logger.warning(
                    "openai_compatible_rate_limit",
                    model_id=self.model_id,
                    status_code=status,
                )
                raise RateLimitError(
                    f"Rate limit exceeded (HTTP 429) for model '{self.model_name}' on '{self.base_url}'."
                )

            # 5xx -> Transient (max 2 retries counted against model's daily budget)
            if 500 <= status < 600:
                if attempt <= max_retries:
                    logger.warning(
                        "openai_compatible_transient_5xx_retrying",
                        model_id=self.model_id,
                        status_code=status,
                        attempt=attempt,
                    )
                    await asyncio.sleep(0.05 * (2 ** (attempt - 1)))
                    continue
                raise LLMProviderError(
                    f"Model '{self.model_name}' server error (HTTP {status}) from '{self.base_url}'."
                )

            # Other 4xx
            raise LLMProviderError(
                f"Model '{self.model_name}' request failed with HTTP {status} from '{self.base_url}'."
            )

    async def generate(
        self, prompt: str, system_prompt: str | None = None, **kwargs: object
    ) -> str:
        """Generate plain text using chat/completions endpoint."""
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0.2,
        }
        data = await self._post_chat(payload)
        choices = data.get("choices")
        if not choices or not isinstance(choices, list):
            raise LLMProviderError(f"No choices returned from '{self.base_url}'")
        msg = choices[0].get("message", {})
        return str(msg.get("content", ""))

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        **kwargs: object,
    ) -> T:
        """
        Generate structured output adhering to schema.
        If supports_json_schema: sends response_format with json_schema.
        Otherwise: sends JSON mode with schema instruction and validates;
        validation errors go through the retry/repair reflection flow.
        """
        import json as _json

        schema_json = schema.model_json_schema()
        messages: list[dict[str, str]] = []

        if self.supports_json_schema:
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            payload: dict[str, object] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": schema_json,
                        "strict": True,
                    },
                },
            }
        else:
            json_instruction = (
                f"Reply with valid JSON only adhering strictly to this schema: {_json.dumps(schema_json)}"
            )
            combined_sys = f"{system_prompt}\n\n{json_instruction}" if system_prompt else json_instruction
            messages.append({"role": "system", "content": combined_sys})
            messages.append({"role": "user", "content": prompt})
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }

        data = await self._post_chat(payload)
        choices = data.get("choices", [])
        if not choices or not isinstance(choices, list):
            raise LLMProviderError(f"No choices returned from '{self.base_url}'")
        content = str(choices[0].get("message", {}).get("content", ""))

        try:
            return schema.model_validate_json(content)
        except Exception as first_exc:
            logger.warning(
                "openai_compatible_structured_validation_failed_repairing",
                model_id=self.model_id,
                schema=schema.__name__,
                error=str(first_exc),
            )
            repair_messages = list(messages)
            repair_messages.append({"role": "assistant", "content": content})
            repair_messages.append({
                "role": "user",
                "content": (
                    f"IMPORTANT: Your previous output failed schema validation with error:\n"
                    f"{first_exc}\n"
                    f"Please correct the error and reply with valid JSON only matching the schema."
                ),
            })
            repair_payload = dict(payload)
            repair_payload["messages"] = repair_messages
            repair_data = await self._post_chat(repair_payload)
            repair_choices = repair_data.get("choices", [])
            if not repair_choices:
                raise LLMProviderError(
                    f"No choices returned during structured repair from '{self.base_url}'"
                )
            repaired_content = str(repair_choices[0].get("message", {}).get("content", ""))
            try:
                return schema.model_validate_json(repaired_content)
            except Exception as repair_exc:
                raise LLMProviderError(
                    f"Schema validation failed after repair for {schema.__name__}: {repair_exc}"
                ) from repair_exc



def _map_gemini_error(exc: Exception, model_name: str) -> LLMProviderError:
    """Map Gemini SDK and HTTP/transport errors to a readable LLMProviderError."""
    if isinstance(exc, LLMProviderError):
        return exc
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        message = getattr(exc, "message", None) or str(exc)
        if code == 429 or "RESOURCE_EXHAUSTED" in str(exc) or "quota" in str(exc).lower():
            return RateLimitError(
                f"Gemini API rate limit exceeded for model '{model_name}': {message}"
            )
        if (
            code in (401, 403)
            or "PERMISSION_DENIED" in str(exc)
            or "UNAUTHENTICATED" in str(exc)
            or "API_KEY_INVALID" in str(exc)
        ):
            return LLMProviderError(
                f"Gemini API authentication failed for model '{model_name}': {message}"
            )
        return LLMProviderError(
            f"Gemini API error ({code}) for model '{model_name}': {message}"
        )
    if isinstance(exc, httpx.HTTPStatusError):
        if exc.response.status_code == 429 or "quota" in exc.response.text.lower():
            return RateLimitError(
                f"Gemini API rate limit exceeded (HTTP 429) for model '{model_name}': {exc.response.text}"
            )
        return LLMProviderError(
            f"Gemini HTTP error ({exc.response.status_code}) for model '{model_name}': {exc}"
        )
    if isinstance(exc, httpx.RequestError):
        return LLMProviderError(
            f"Gemini network error for model '{model_name}': {exc}"
        )
    exc_lower = str(exc).lower()
    if "429" in exc_lower or "quota" in exc_lower or "resource_exhausted" in exc_lower:
        return RateLimitError(
            f"Gemini API rate limit exceeded for model '{model_name}': {exc}"
        )
    return LLMProviderError(
        f"Gemini request failed for model '{model_name}': {exc}"
    )


def _extract_retry_after(exc: Exception) -> float | None:
    """Extract retry-after in seconds from exception headers or message."""
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "headers") and resp.headers:
        ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
        if ra:
            try:
                return float(ra)
            except (ValueError, TypeError):
                pass
    text = str(exc)
    m = re.search(r"retry[-_\s]after[:\s]+(\d+(?:\.\d+)?)", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    m = re.search(r"reset\s+after\s+(\d+(?:\.\d+)?)\s*s", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    m = re.search(r"wait\s+(\d+(?:\.\d+)?)\s*s", text, re.I)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _is_5xx_error(exc: Exception) -> bool:
    """Check if exception represents a transient 5xx server error."""
    if isinstance(exc, errors.ServerError):
        return True
    code = getattr(exc, "code", None)
    if isinstance(code, int) and 500 <= code < 600:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "status_code") and isinstance(resp.status_code, int):
        if 500 <= resp.status_code < 600:
            return True
    if isinstance(exc, httpx.HTTPStatusError) and 500 <= exc.response.status_code < 600:
        return True
    text = str(exc).lower()
    if "500 internal" in text or "502 bad gateway" in text or "503 service" in text or "504 gateway" in text:
        return True
    return False


def _is_429_error(exc: Exception, mapped: LLMProviderError | None = None) -> bool:
    """Check if exception represents a 429 rate limit or quota exhaustion."""
    if isinstance(mapped, RateLimitError):
        return True
    code = getattr(exc, "code", None)
    if code == 429:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None and hasattr(resp, "status_code") and resp.status_code == 429:
        return True
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
        return True
    text = str(exc).lower()
    if "429" in text or "quota" in text or "resource_exhausted" in text or "rate limit" in text:
        return True
    return False


class GeminiProvider:
    """Gemini model provider using official google-genai Python SDK."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        client: genai.Client | None = None,
        thinking_budget: int | None = None,
        timeout: float | None = None,
        rate_gate: object | None = None,
        model_id: str = "gemini-main",
    ):
        cfg = None
        try:
            from app.config import get_settings

            cfg = get_settings()
        except Exception:
            pass

        model = (
            model_name
            or os.getenv("GEMINI_MODEL")
            or (cfg.gemini_model if cfg else None)
        )
        if not model:
            raise LLMProviderError(
                "GEMINI_MODEL is not set. Please configure GEMINI_MODEL in .env."
            )
        self.model_name = model
        self.model_id = model_id
        self.rate_gate = rate_gate

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
            or (cfg.gemini_api_key if cfg else None)
        )
        if not self.api_key and client is None:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in .env."
            )

        if thinking_budget is not None:
            self.thinking_budget = thinking_budget
        elif cfg and hasattr(cfg, "gemini_thinking_budget"):
            self.thinking_budget = cfg.gemini_thinking_budget
        else:
            self.thinking_budget = 0

        if timeout is not None:
            self.timeout = float(timeout)
        elif cfg and hasattr(cfg, "gemini_timeout_seconds"):
            self.timeout = float(cfg.gemini_timeout_seconds)
        elif cfg and hasattr(cfg, "gemini_timeout"):
            self.timeout = float(cfg.gemini_timeout)
        else:
            self.timeout = 60.0

        self.last_response: types.GenerateContentResponse | None = None
        self.last_usage_metadata: types.GenerateContentResponseUsageMetadata | None = None

        timeout_ms = int(self.timeout * 1000)
        self._http_options = types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        )

        if client is not None:
            self._client = client
        elif self.api_key:
            self._client = genai.Client(api_key=self.api_key, http_options=self._http_options)
        else:
            self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise LLMProviderError(
                "GEMINI_API_KEY is not set. Please configure GEMINI_API_KEY in .env."
            )
        timeout_ms = int(self.timeout * 1000)
        self._http_options = types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        )
        self._client = genai.Client(api_key=self.api_key, http_options=self._http_options)
        return self._client

    def get_client(self) -> genai.Client:
        """Return the underlying genai.Client instance."""
        return self._get_client()

    def _build_thinking_config(
        self,
        thinking_budget: int | None | object = ...,
        thinking_level: str | types.ThinkingLevel | None = None,
    ) -> types.ThinkingConfig | None:
        model_lower = (self.model_name or "").lower()
        # Set thinking_level (low) for models that use it (e.g. Gemini 3 series)
        if "gemini-3" in model_lower or "3." in model_lower:
            lvl = thinking_level or types.ThinkingLevel.LOW
            if isinstance(lvl, str):
                lvl_clean = lvl.upper()
                lvl = getattr(types.ThinkingLevel, lvl_clean, types.ThinkingLevel.LOW)
            return types.ThinkingConfig(thinking_level=lvl)

        # Set thinking_budget for models that use a budget (e.g. Gemini 2.5 series)
        budget = self.thinking_budget if thinking_budget is ... else thinking_budget
        if budget is not None:
            return types.ThinkingConfig(thinking_budget=int(budget))
        return None

    async def _execute_with_retry(
        self,
        call_fn: Callable[[], object],
        max_5xx_retries: int = 2,
        max_retries: int | None = None,
    ) -> object:
        """
        Execute an async Gemini API call.
        - Every call goes through global Gemini rate gate (sliding window RPM + daily budget).
        - No automatic retries on 429 or quota errors (fails immediately).
        - Transient 503 gets at most 2 retries with backoff, each counted against daily budget.
        """
        if max_retries is not None:
            max_5xx_retries = max_retries

        from app.llm.rate_gate import get_rate_gate

        gate = getattr(self, "rate_gate", None) or get_rate_gate()
        delay = 1.0
        attempt_5xx = 0

        while True:
            # Pass through the rate gate before each call (including 5xx retries)
            if hasattr(gate, "acquire"):
                try:
                    await gate.acquire(self.model_id)
                except TypeError:
                    await gate.acquire()
            elif asyncio.iscoroutinefunction(gate):
                await gate()

            try:
                return await asyncio.wait_for(call_fn(), timeout=self.timeout)
            except TimeoutError as exc:
                raise LLMProviderError(f"Gemini call timed out after {self.timeout}s") from exc
            except Exception as exc:
                mapped = _map_gemini_error(exc, self.model_name)

                # Transient 5xx retry: up to max_5xx_retries with exponential backoff
                if _is_5xx_error(exc):
                    if attempt_5xx < max_5xx_retries:
                        attempt_5xx += 1
                        jitter = random.uniform(0.1, 0.5)
                        sleep_time = delay + jitter
                        logger.warning(
                            "gemini_5xx_transient_backing_off",
                            attempt=attempt_5xx,
                            sleep_seconds=round(sleep_time, 2),
                            error=str(exc),
                        )
                        await asyncio.sleep(sleep_time)
                        delay *= 2.0
                        continue
                    else:
                        raise mapped from exc

                # 429 RateLimit / Quota: NO automatic retries -> fail immediately!
                if _is_429_error(exc, mapped):
                    raise mapped from exc

                # Any other error -> fail immediately
                raise mapped from exc

    async def generate_with_search(
        self, prompt: str, thinking_budget: int | None = None, **kwargs: object
    ) -> types.GenerateContentResponse:
        """
        Generate content using Gemini with Google Search tool enabled.
        Plain text call only; does not use response_schema or JSON mode.
        """
        client = self._get_client()
        budget = thinking_budget if thinking_budget is not None else kwargs.get("thinking_budget")
        tc = self._build_thinking_config(budget)
        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            thinking_config=tc,
        )

        async def _call() -> types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

        response = await self._execute_with_retry(_call)
        self.last_response = response
        self.last_usage_metadata = getattr(response, "usage_metadata", None)
        thoughts_tokens = getattr(self.last_usage_metadata, "thoughts_token_count", None)
        if thoughts_tokens is not None:
            logger.info("gemini_thoughts_token_count", model=self.model_name, thoughts_token_count=thoughts_tokens)
        return response

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        thinking_budget: int | None = None,
        **kwargs: object,
    ) -> str:
        """Generate plain text from prompt using Gemini model."""
        client = self._get_client()
        budget = thinking_budget if thinking_budget is not None else kwargs.get("thinking_budget")
        tc = self._build_thinking_config(budget)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=tc,
        )

        async def _call() -> types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

        response = await self._execute_with_retry(_call)
        self.last_response = response
        self.last_usage_metadata = getattr(response, "usage_metadata", None)
        thoughts_tokens = getattr(self.last_usage_metadata, "thoughts_token_count", None)
        if thoughts_tokens is not None:
            logger.info("gemini_thoughts_token_count", model=self.model_name, thoughts_token_count=thoughts_tokens)
        return getattr(response, "text", "") or ""

    async def generate_structured(
        self,
        prompt: str,
        schema: type[T],
        system_prompt: str | None = None,
        thinking_budget: int | None = None,
        **kwargs: object,
    ) -> T:
        """Generate structured output adhering to schema using Gemini model."""
        client = self._get_client()
        budget = thinking_budget if thinking_budget is not None else kwargs.get("thinking_budget")
        tc = self._build_thinking_config(budget)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=tc,
        )

        async def _call() -> types.GenerateContentResponse:
            return await client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

        response = await self._execute_with_retry(_call)
        self.last_response = response
        self.last_usage_metadata = getattr(response, "usage_metadata", None)
        thoughts_tokens = getattr(self.last_usage_metadata, "thoughts_token_count", None)
        if thoughts_tokens is not None:
            logger.info("gemini_thoughts_token_count", model=self.model_name, thoughts_token_count=thoughts_tokens)
        text = getattr(response, "text", "") or ""

        return schema.model_validate_json(text)



def get_llm_provider(
    provider_name: str | None = None,
    model_name: str | None = None,
    model_id: str | None = None,
    **kwargs: object,
) -> LLMProvider:
    """
    Factory function returning the configured LLMProvider.
    Selected via provider_name or LLM_PROVIDER env var ('ollama' | 'gemini').
    """
    cfg = None
    try:
        from app.config import get_settings

        cfg = get_settings()
    except Exception:
        pass

    chosen = (
        provider_name
        or os.getenv("LLM_PROVIDER")
        or (cfg.llm_provider if cfg else "ollama")
    ).lower().strip()

    base_url = kwargs.pop("base_url", None)
    api_key_env = kwargs.pop("api_key_env", None)
    supports_json_schema = bool(kwargs.pop("supports_json_schema", False))

    if chosen == "gemini":
        mid = model_id or "gemini-main"
        return GeminiProvider(model_name=model_name, model_id=mid, **kwargs)
    elif chosen == "ollama":
        mid = model_id or "ollama-local"
        return OllamaProvider(model_name=model_name, model_id=mid, **kwargs)
    elif chosen == "openai_compatible":
        mid = model_id or "openai-compatible"
        return OpenAICompatibleProvider(
            base_url=str(base_url or ""),
            api_key_env=api_key_env,
            model_name=model_name,
            model_id=mid,
            supports_json_schema=supports_json_schema,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported LLM_PROVIDER '{chosen}'. Supported options are 'gemini', 'ollama', and 'openai_compatible'."
        )

