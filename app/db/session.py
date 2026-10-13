"""
Database session factory.

TODO (next task): implement async SQLAlchemy engine, session factory,
and pgvector-enabled Base for ORM models.
"""
from app.config import get_settings


def get_database_url() -> str:
    """Return the async database URL from settings."""
    return get_settings().database_url
