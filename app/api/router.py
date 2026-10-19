"""
API router – aggregates all route modules.
"""
from fastapi import APIRouter

from app.api.health import router as health_router
from app.api.learn import router as learn_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(learn_router, prefix="/api", tags=["learning"])
api_router.include_router(learn_router, tags=["learning"])

