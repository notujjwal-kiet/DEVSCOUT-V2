"""
pytest configuration and shared fixtures.
"""
import pytest
from app.config import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache():
    """Reset the lru_cache on get_settings() before/after each test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
