"""
/health endpoint.

Returns the app version, environment, and configured Ollama model so
callers can verify the service is reachable and properly configured.
"""
from datetime import UTC, datetime

from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    ollama_host: str
    ollama_model: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Lightweight liveness probe – no database required."""
    cfg = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=cfg.app_env,
        ollama_host=cfg.ollama_host,
        ollama_model=cfg.ollama_model,
        timestamp=datetime.now(UTC).isoformat(),
    )
