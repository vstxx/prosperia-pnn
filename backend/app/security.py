from __future__ import annotations

import hmac
from typing import Annotated, Optional

from fastapi import Header, HTTPException, status

from .config import get_settings


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def require_internal_api_key(
    authorization: Annotated[Optional[str], Header()] = None,
    x_pnn_secret: Annotated[Optional[str], Header()] = None,
) -> None:
    settings = get_settings()
    expected = settings.internal_api_key or settings.pnn_shared_secret
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="INTERNAL_API_KEY is not configured",
        )

    received = _bearer_token(authorization)
    if received and hmac.compare_digest(received, expected):
        return

    if x_pnn_secret and settings.pnn_shared_secret and hmac.compare_digest(x_pnn_secret, settings.pnn_shared_secret):
        return

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid internal API key",
    )


def require_shared_secret(
    authorization: Annotated[Optional[str], Header()] = None,
    x_pnn_secret: Annotated[Optional[str], Header()] = None,
) -> None:
    settings = get_settings()
    expected = settings.pnn_shared_secret or settings.internal_api_key
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PNN_SHARED_SECRET is not configured",
        )

    received = x_pnn_secret or _bearer_token(authorization)
    if not received or not hmac.compare_digest(received, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid shared secret",
        )
