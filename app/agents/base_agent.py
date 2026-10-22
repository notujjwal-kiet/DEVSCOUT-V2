"""
Strands Agents – base agent configuration.
"""
from app.config import get_settings


def get_agent_config() -> dict:
    """Return minimal agent configuration from settings."""
    cfg = get_settings()
    return {
        "provider": "gemini",
        "model": cfg.gemini_model,
        "timeout": cfg.gemini_timeout,
    }
