"""
DEVSCOUT – Configuration loader.

All settings are read from environment variables (or a .env file loaded
by python-dotenv before this module is imported).  No secrets are ever
hardcoded here.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings object – one instance is shared across the app."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ── App ───────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── LLM Provider ──────────────────────────────────────────────
    llm_provider: str = Field(default="gemini")
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_timeout_seconds: float = Field(default=60.0, ge=1.0)
    gemini_timeout: float = Field(default=60.0, ge=1.0)
    gemini_rpm: int = Field(default=4, ge=1)
    gemini_rpd_limit: int = Field(default=18, ge=1)
    gemini_quota_file: str = Field(default="data/quota.json")
    request_cache_file: str = Field(default="data/responses.json")
    gemini_thinking_budget: int = Field(default=0, ge=0)
    model_pricing_per_1m: dict[str, dict[str, float]] = Field(
        default_factory=lambda: {
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
            "gemini-2.5-pro": {"input": 1.25, "output": 5.00},
        }
    )

    # ── Seed Documentation Fallback ──────────────────────────────
    seed_docs: dict[str, str] = Field(
        default_factory=lambda: {
            "fastapi": "https://fastapi.tiangolo.com/tutorial/",
            "pydantic": "https://docs.pydantic.dev/latest/",
            "python": "https://docs.python.org/3/tutorial/",
            "react": "https://react.dev/learn",
            "typescript": "https://www.typescriptlang.org/docs/",
        }
    )

    # ── PostgreSQL ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://devscout:devscout_secret@localhost:5432/devscout"
    )

    # ── Evidence / research ───────────────────────────────────────
    evidence_cache_ttl_seconds: int = Field(default=86400, ge=0)
    research_cache_file: str = Field(default="data/research_cache.json")
    research_cache_dir: str = Field(default="data/research_cache")
    web_research_mode: Literal["live", "cached"] = "cached"
    github_token: str | None = Field(default=None)

    # ── Pipeline tuning ───────────────────────────────────────────
    max_evidence_retry_loops: int = Field(default=1, ge=1, le=5)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    answer_coverage_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    min_questions_answered: int = Field(default=2, ge=1)
    min_sources_per_topic: int = Field(default=2, ge=1)
    min_distinct_domains: int = Field(default=2, ge=1)
    min_distinct_source_types: int = Field(default=2, ge=1)
    require_official_or_repo: bool = Field(default=True)
    freshness_stale_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    max_topics_per_run: int = Field(default=2, ge=1)
    max_searches_per_run: int = Field(default=15, ge=1)
    enable_github_queries: bool = Field(default=False)
    enable_community_queries: bool = Field(default=False)

    # ── Excerpts & Evidence Grounding ─────────────────────────────
    max_excerpt_chars: int = Field(default=4000, ge=100)
    stage5_excerpt_budget_chars: int = Field(default=12000, ge=1000)
    stage5_top_n_resources: int = Field(default=3, ge=1)
    support_token_overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

    # ── Readiness & Attempt Review ─────────────────────────────────
    snippet_copy_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    min_concept_references_for_relevance: int = Field(default=2, ge=0)
    max_understanding_attempts: int = Field(default=5, ge=1)
    attempt_cooldown_seconds: float = Field(default=30.0, ge=0.0)

    # ── Jobs, Security & Session Limits ───────────────────────────
    pipeline_timeout_seconds: float = Field(default=480.0, ge=1.0)
    api_keys: str = Field(default="")
    rate_limit_capacity: float = Field(default=10.0, ge=1.0)
    rate_limit_refill_rate: float = Field(default=2.0, ge=0.01)
    max_request_size_bytes: int = Field(default=1_048_576, ge=1024)
    session_ttl_seconds: float = Field(default=86400.0, ge=1.0)
    max_sessions: int = Field(default=1000, ge=1)

    # ── CORS ──────────────────────────────────────────────────────
    cors_origins: list[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v: object) -> list[str]:
        if isinstance(v, str):
            return [orig.strip() for orig in v.split(",") if orig.strip()]
        if isinstance(v, list):
            return [str(orig).strip() for orig in v if str(orig).strip()]
        return ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("log_level")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


def calculate_token_cost(
    model_name: str | None,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    is_local: bool = False,
    model_id: str | None = None,
) -> float | None:
    """
    Calculate USD cost based on per-model registry pricing or config fallback table.
    Rules:
    - Local models -> 0.0
    - If model is in registry and has price_in_per_1m / price_out_per_1m -> calculated cost
    - If model is in config pricing table -> calculated cost
    - Unknown cost -> None
    """
    if is_local:
        return 0.0
    if not model_name and not model_id:
        return None
    model_lower = str(model_name or "").strip().lower()

    if prompt_tokens is None and completion_tokens is None:
        return None

    p_tokens = prompt_tokens or 0
    c_tokens = completion_tokens or 0

    # 1. Check registry model entry pricing
    try:
        from app.llm.registry import get_model_registry

        reg = get_model_registry()
        entry = reg.get_model(model_id) if model_id else None
        if entry is None and model_name:
            for m in reg.list_models():
                if m.model.lower() == model_lower or m.id.lower() == model_lower:
                    entry = m
                    break
        if entry is not None:
            if entry.local:
                return 0.0
            if entry.price_in_per_1m is not None or entry.price_out_per_1m is not None:
                in_rate = entry.price_in_per_1m or 0.0
                out_rate = entry.price_out_per_1m or 0.0
                cost = (p_tokens * in_rate + c_tokens * out_rate) / 1_000_000.0
                return round(cost, 6)
    except Exception:
        pass

    # 2. Check config table fallback
    pricing = get_settings().model_pricing_per_1m
    model_rates = None
    for k, v in pricing.items():
        if k.lower() == model_lower or model_lower.startswith(k.lower()):
            model_rates = v
            break

    if model_rates is None:
        return None

    input_rate = model_rates.get("input", 0.0)
    output_rate = model_rates.get("output", 0.0)

    cost = (p_tokens * input_rate + c_tokens * output_rate) / 1_000_000.0
    return round(cost, 6)



SEED_DOCS: dict[str, str] = {
    "fastapi": "https://fastapi.tiangolo.com/tutorial/",
    "pydantic": "https://docs.pydantic.dev/latest/",
    "python": "https://docs.python.org/3/tutorial/",
    "react": "https://react.dev/learn",
    "typescript": "https://www.typescriptlang.org/docs/",
}


