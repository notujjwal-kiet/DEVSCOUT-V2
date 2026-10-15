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
    gemini_model: str = Field(default="gemini-1.5-flash")

    # ── PostgreSQL ────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://devscout:devscout_secret@localhost:5432/devscout"
    )

    # ── Evidence / research ───────────────────────────────────────
    evidence_cache_ttl_seconds: int = Field(default=86400, ge=0)
    web_research_mode: Literal["live", "cached"] = "cached"

    # ── Pipeline tuning ───────────────────────────────────────────
    max_evidence_retry_loops: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    @field_validator("log_level")
    @classmethod
    def _uppercase_log_level(cls, v: str) -> str:
        return v.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
