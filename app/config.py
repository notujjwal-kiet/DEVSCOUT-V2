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
    )

    # ── App ───────────────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    # ── LLM Provider ──────────────────────────────────────────────
    llm_provider: str = Field(default="ollama")
    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    ollama_timeout: int = Field(default=120, ge=10)
    ollama_think: bool = Field(default=False)
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str | None = Field(default=None)

    # ── PostgreSQL ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://devscout:devscout_secret@localhost:5432/devscout"
    )

    # ── Evidence / research ───────────────────────────────────────
    evidence_cache_ttl_seconds: int = Field(default=86400, ge=0)
    web_research_mode: Literal["live", "cached"] = "cached"
    github_token: str | None = Field(default=None)

    # ── Pipeline tuning ───────────────────────────────────────────
    max_evidence_retry_loops: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    answer_coverage_threshold: float = Field(default=0.6, ge=0.0, le=1.0)

    # ── Excerpts & Evidence Grounding ─────────────────────────────
    max_excerpt_chars: int = Field(default=4000, ge=100)
    stage5_excerpt_budget_chars: int = Field(default=12000, ge=1000)
    stage5_top_n_resources: int = Field(default=3, ge=1)
    support_token_overlap_threshold: float = Field(default=0.15, ge=0.0, le=1.0)

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


# ── Configuration Constants ───────────────────────────────────────────────────
MAX_EXCERPT_CHARS: int = 4000
STAGE5_EXCERPT_BUDGET_CHARS: int = 12000
STAGE5_TOP_N_RESOURCES: int = 3
SUPPORT_TOKEN_OVERLAP_THRESHOLD: float = 0.15
ANSWER_COVERAGE_THRESHOLD: float = 0.6



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

