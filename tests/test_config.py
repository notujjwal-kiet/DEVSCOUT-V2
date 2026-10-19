"""
Tests for app/config.py – Settings loader.
"""
import os

import pytest

from app.config import Settings, get_settings


def test_default_settings(monkeypatch):
    """Settings instantiate with sensible defaults without a .env file."""
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    s = Settings()
    assert s.app_env == "development"
    assert s.app_port == 8000
    assert "11434" in s.ollama_host
    assert s.ollama_model != ""
    assert 0.0 < s.confidence_threshold <= 1.0
    assert s.max_evidence_retry_loops >= 1


def test_log_level_uppercase():
    """log_level is always stored in uppercase."""
    s = Settings(log_level="debug")
    assert s.log_level == "DEBUG"


def test_get_settings_returns_singleton():
    """get_settings() returns the same cached object on repeated calls."""
    # Clear the lru_cache to isolate this test.
    get_settings.cache_clear()
    a = get_settings()
    b = get_settings()
    assert a is b
    get_settings.cache_clear()


def test_ollama_model_from_env(monkeypatch):
    """OLLAMA_MODEL env var overrides the default."""
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    s = Settings()
    assert s.ollama_model == "mistral"


def test_settings_caps_and_evidence_ttl(monkeypatch):
    """Settings include max_topics_per_run, max_searches_per_run, evidence_cache_ttl_seconds."""
    s = Settings()
    assert s.max_topics_per_run >= 1
    assert s.max_searches_per_run >= 1
    assert s.evidence_cache_ttl_seconds >= 1

    monkeypatch.setenv("MAX_TOPICS_PER_RUN", "7")
    monkeypatch.setenv("MAX_SEARCHES_PER_RUN", "25")
    s2 = Settings()
    assert s2.max_topics_per_run == 7
    assert s2.max_searches_per_run == 25


def test_no_duplicated_module_level_constants_in_config():
    """Verify deprecated module-level constants are removed from app.config; consumers read get_settings()."""
    import app.config as config_mod

    removed_constants = [
        "MAX_EXCERPT_CHARS",
        "STAGE5_EXCERPT_BUDGET_CHARS",
        "STAGE5_TOP_N_RESOURCES",
        "SUPPORT_TOKEN_OVERLAP_THRESHOLD",
        "DEFAULT_MIN_DISTINCT_DOMAINS",
        "DEFAULT_MIN_DISTINCT_SOURCE_TYPES",
        "DEFAULT_FRESHNESS_STALE_THRESHOLD",
        "REQUIRE_OFFICIAL_DOCS_OR_REPO",
        "MIN_SOURCES_PER_TOPIC",
        "ANSWER_COVERAGE_THRESHOLD",
    ]
    for const_name in removed_constants:
        assert not hasattr(config_mod, const_name), f"{const_name} should not be defined at module level in app.config"

    cfg = get_settings()
    for const_name in removed_constants:
        if const_name == "REQUIRE_OFFICIAL_DOCS_OR_REPO":
            attr_name = "require_official_or_repo"
        else:
            attr_name = const_name.lower().removeprefix("default_")
        assert hasattr(cfg, attr_name), f"Settings should have {attr_name}"
