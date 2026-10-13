"""
Strands Agents – base agent placeholder.

TODO (next task): implement the Strands Agent wrapper that targets the
configured Ollama provider and exposes tool interfaces for web research.
"""
from app.config import get_settings


def get_agent_config() -> dict:
    """Return minimal agent configuration from settings."""
    cfg = get_settings()
    return {
        "provider": "ollama",
        "host": cfg.ollama_host,
        "model": cfg.ollama_model,
        "timeout": cfg.ollama_timeout,
    }
