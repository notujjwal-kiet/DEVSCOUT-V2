"""
app/llm/registry.py

Model registry loaded from models.json at repo root:
- Validated with Pydantic.
- Unknown providers fail startup with a clear message.
- Model names resolve environment variables (${GEMINI_MODEL}, ${OLLAMA_MODEL}) dynamically.
- No secrets stored.
"""
from __future__ import annotations

import json
import os
import re
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator
import structlog

logger = structlog.get_logger(__name__)

MODELS_FILE_DEFAULT = "models.json"


def resolve_model_name(val: str, provider: str) -> str:
    """Resolve environment variable placeholder in model string if present."""
    if not val:
        return "gemini-2.5-flash" if provider == "gemini" else "llama3.2"

    raw = val.strip()
    m = re.match(r"^\$\{?([A-Za-z0-9_]+)\}?$", raw)
    if m:
        var_name = m.group(1)
        env_val = os.getenv(var_name)
        if env_val:
            return env_val
        try:
            from app.config import get_settings

            cfg = get_settings()
            cfg_val = getattr(cfg, var_name.lower(), None)
            if cfg_val:
                return str(cfg_val)
        except Exception:
            pass
        return "gemini-2.5-flash" if provider == "gemini" else "llama3.2"
    return raw


class ModelEntry(BaseModel):
    """Configuration for a single registered model."""

    id: str = Field(..., description="Unique model identifier")
    label: str = Field(..., description="Human-readable label")
    provider: Literal["gemini", "ollama", "openai_compatible"] = Field(..., description="Provider type")
    model: str = Field(..., description="Model name or environment variable template")
    local: bool = Field(default=False, description="Whether inference runs locally")
    base_url: str | None = Field(default=None, description="Base URL for openai_compatible provider (https or localhost http)")
    api_key_env: str | None = Field(default=None, description="Environment variable name for the API key")
    supports_json_schema: bool = Field(default=False, description="Whether model supports structured JSON schema response_format")
    rpm: int | None = Field(default=None, description="Requests per minute limit")
    rpd: int | None = Field(default=None, description="Requests per day limit")
    price_in_per_1m: float | None = Field(default=None, description="Input price per 1M tokens in USD")
    price_out_per_1m: float | None = Field(default=None, description="Output price per 1M tokens in USD")
    tier: Literal["stable", "experimental"] = Field(default="stable", description="Release tier")
    priority: int = Field(default=1, description="Selection priority (lowest number chosen first)")

    @field_validator("provider", mode="before")
    @classmethod
    def _validate_provider(cls, v: object) -> str:
        s = str(v).lower().strip()
        if s not in ("gemini", "ollama", "openai_compatible"):
            raise ValueError(
                f"Unknown/unsupported provider '{v}'. Supported providers are 'gemini', 'ollama', and 'openai_compatible'."
            )
        return s

    @field_validator("tier", mode="before")
    @classmethod
    def _validate_tier(cls, v: object) -> str:
        s = str(v).lower().strip()
        if s not in ("stable", "experimental"):
            raise ValueError(
                f"Unknown tier '{v}'. Supported tiers are 'stable' and 'experimental'."
            )
        return s

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        url = str(v).strip()
        if not url:
            return None
        from urllib.parse import urlparse

        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        hostname = (parsed.hostname or "").lower()

        if scheme == "http":
            if hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError(
                    f"base_url '{url}' must use HTTPS. HTTP is only allowed for localhost entries."
                )
        elif scheme != "https":
            raise ValueError(
                f"base_url '{url}' has invalid scheme '{scheme}'. Only HTTPS (and HTTP for localhost) are allowed."
            )
        return url

    @model_validator(mode="after")
    def _validate_entry(self) -> ModelEntry:
        self.model = resolve_model_name(self.model, self.provider)
        if self.provider == "openai_compatible":
            if not self.base_url:
                raise ValueError("`base_url` is required for provider 'openai_compatible'.")
        return self


class ModelRegistry:
    """In-memory registry of models loaded from models.json."""

    def __init__(self, models: list[ModelEntry] | None = None) -> None:
        self._models: dict[str, ModelEntry] = {}
        if models:
            for m in models:
                self._models[m.id] = m

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self._models.get(model_id)

    def has_model(self, model_id: str) -> bool:
        return model_id in self._models

    def list_models(self) -> list[ModelEntry]:
        """Return registered models sorted by priority ascending (lowest number first)."""
        return sorted(self._models.values(), key=lambda x: x.priority)

    def get_model_ids(self) -> list[str]:
        return list(self._models.keys())

    def get_primary_model(self) -> ModelEntry | None:
        models = self.list_models()
        return models[0] if models else None


_registry: ModelRegistry | None = None


def load_model_registry(file_path: str = MODELS_FILE_DEFAULT) -> ModelRegistry:
    """
    Load and validate models.json using Pydantic.
    Fails startup with a clear message if unknown providers are present.
    """
    global _registry
    if not os.path.exists(file_path):
        # Fallback to repo root if path is relative and not found in cwd
        repo_root_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), file_path)
        if os.path.exists(repo_root_path):
            file_path = repo_root_path
        else:
            raise FileNotFoundError(f"Model registry file '{file_path}' not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Strip C-style /* ... */ and // comments for example/annotated JSON files
    clean_json = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
    clean_json = re.sub(r"//.*$", "", clean_json, flags=re.MULTILINE)
    raw = json.loads(clean_json)

    entries_raw = raw if isinstance(raw, list) else raw.get("models", list(raw.values()))
    models: list[ModelEntry] = []
    for item in entries_raw:
        try:
            entry = ModelEntry.model_validate(item)
            models.append(entry)
        except Exception as exc:
            msg = f"Failed to validate model entry in {file_path}: {exc}"
            logger.error("model_registry_validation_failed", error=msg)
            raise ValueError(msg) from exc

    _registry = ModelRegistry(models)
    logger.info("model_registry_loaded", model_count=len(models), models=[m.id for m in models])
    return _registry


def is_model_available(entry: ModelEntry) -> bool:
    """
    Check if a model is currently available:
    - openai_compatible: requires api_key_env to be set in os.environ (or local Ollama entry) AND has budget.
    - gemini: requires GEMINI_API_KEY (if remote) AND has budget.
    - ollama: local models are available if server responds/configured.
    """
    from app.llm.rate_gate import get_rate_gate

    gate = get_rate_gate()
    if not gate.has_budget(entry.id):
        return False

    if entry.provider == "openai_compatible":
        if entry.local:
            return True
        if not entry.api_key_env:
            return False
        return bool(os.getenv(entry.api_key_env, "").strip())

    if entry.provider == "gemini":
        if not entry.local:
            return bool(os.getenv("GEMINI_API_KEY", "").strip())
        return True

    if entry.provider == "ollama":
        return True

    return False


def get_model_registry() -> ModelRegistry:
    """Return singleton ModelRegistry instance, loading models.json if needed."""
    global _registry
    if _registry is None:
        _registry = load_model_registry()
    return _registry


def set_model_registry(registry: ModelRegistry | None) -> None:
    """Override singleton ModelRegistry (used in unit tests)."""
    global _registry
    _registry = registry

