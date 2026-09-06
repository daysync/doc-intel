"""Readiness: can this instance actually serve? Liveness stays at /health."""

from typing import Annotated, Any

import httpx2 as httpx
from fastapi import APIRouter, Depends, Request, Response, status

from doc_intel.api.security import get_request_settings
from doc_intel.api.settings import Settings

router = APIRouter()


async def _postgres_ok(request: Request) -> tuple[bool, str]:
    pool = getattr(request.app.state, "pool", None)
    if pool is None:
        return True, "in-memory store"
    try:
        async with pool.connection() as connection:
            await connection.execute("SELECT 1")
        return True, "ok"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"[:200]


async def _model_host_ok(settings: Settings, provider: str) -> tuple[bool, str]:
    if provider != "ollama":
        return True, f"{provider}: not probed"
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{settings.ollama_host.rstrip('/')}/api/tags")
        return response.status_code == 200, f"ollama {response.status_code}"
    except Exception as error:
        return False, f"ollama unreachable: {type(error).__name__}"


@router.get("/ready")
async def ready(
    request: Request,
    response: Response,
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> dict[str, Any]:
    provider = getattr(request.app.state, "llm_provider", None) or settings.llm_provider
    db_ok, db_detail = await _postgres_ok(request)
    model_ok, model_detail = await _model_host_ok(settings, provider)
    ok = db_ok and model_ok
    response.status_code = status.HTTP_200_OK if ok else status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "ready": ok,
        "checks": {"postgres": db_detail, "model_host": model_detail},
        "auth": "api-key" if settings.api_keys else "open",
        "provider": provider,
    }
