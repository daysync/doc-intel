"""API-key authentication.

``API_KEYS`` holds one or more comma-separated keys. When it is empty the API is open,
which is right for a laptop and wrong for anything reachable from the internet; the
readiness endpoint reports which mode is active. Keys are compared in constant time.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from doc_intel.api.settings import Settings, get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _configured_keys(settings: Settings) -> list[str]:
    if settings.api_keys is None:
        return []
    return [k.strip() for k in settings.api_keys.get_secret_value().split(",") if k.strip()]


def get_request_settings(request: Request) -> Settings:
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings or get_settings()


async def require_api_key(
    key: Annotated[str | None, Depends(api_key_header)],
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> None:
    configured = _configured_keys(settings)
    if not configured:
        return
    if key is not None and any(secrets.compare_digest(key, valid) for valid in configured):
        return
    raise HTTPException(status.HTTP_401_UNAUTHORIZED, "missing or invalid API key")
