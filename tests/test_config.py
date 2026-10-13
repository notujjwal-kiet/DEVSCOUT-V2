"""
Tests for app/config.py – Settings loader.
"""
import os

import pytest

from app.config import Settings, get_settings


def test_default_settings():
    """Settings instantiate with sensible defaults without a .env file."""
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
