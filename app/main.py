"""
DEVSCOUT – FastAPI application factory.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.router import api_router
from app.api.middleware import RequestSizeLimitMiddleware

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Startup / shutdown lifecycle."""
    cfg = get_settings()
    from app.llm.registry import load_model_registry

    try:
        registry = load_model_registry()
        logger.info(
            "devscout_models_registered",
            models=registry.get_model_ids(),
        )
    except Exception as exc:
        logger.error("model_registry_startup_failed", error=str(exc))
        raise

    logger.info(
        "devscout_startup",
        env=cfg.app_env,
        ollama_host=cfg.ollama_host,
        ollama_model=cfg.ollama_model,
    )
    yield
    logger.info("devscout_shutdown")


def create_app() -> FastAPI:
    cfg = get_settings()

    application = FastAPI(
        title="DEVSCOUT",
        description="AI-powered technical decision intelligence platform.",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if cfg.app_env == "development" else None,
        redoc_url="/redoc" if cfg.app_env == "development" else None,
    )

    application.add_middleware(RequestSizeLimitMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    return application



app = create_app()
