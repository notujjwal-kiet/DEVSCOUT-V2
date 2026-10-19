"""
app/api/middleware.py

Middleware for enforcing request constraints such as maximum payload size.
"""
from __future__ import annotations

from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.config import get_settings


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject requests whose payload exceeds settings.max_request_size_bytes with 413."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        cfg = get_settings()
        content_length_header = request.headers.get("content-length")

        if content_length_header:
            try:
                content_length = int(content_length_header)
                if content_length > cfg.max_request_size_bytes:
                    return JSONResponse(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        content={"detail": "Request entity too large"},
                    )
            except ValueError:
                pass

        return await call_next(request)
