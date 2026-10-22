"""
/health endpoint.

Returns the app version, environment, and configured LLM provider and model so
callers can verify the service is reachable and properly configured.
"""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.auth import verify_api_key_and_rate_limit
from app.config import get_settings

router = APIRouter(dependencies=[Depends(verify_api_key_and_rate_limit)])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    provider: str
    gemini_model: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Lightweight liveness probe – no database required."""
    cfg = get_settings()
    return HealthResponse(
        status="ok",
        version="0.1.0",
        environment=cfg.app_env,
        provider=cfg.llm_provider,
        gemini_model=cfg.gemini_model,
        timestamp=datetime.now(UTC).isoformat(),
    )
