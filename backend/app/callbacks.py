from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import Settings


logger = logging.getLogger("pnn.backend.callbacks")


@dataclass(frozen=True)
class CallbackResult:
    attempted: bool
    ok: bool
    url: str
    status_code: int | None = None
    body: str | None = None
    error: str | None = None


def send_draft_callback(draft: dict, settings: Settings) -> CallbackResult:
    url = (settings.bot_review_endpoint or settings.pnn_bot_callback_url).strip()
    if not url:
        logger.warning("PNN bot callback skipped: BOT_REVIEW_ENDPOINT is empty")
        return CallbackResult(attempted=False, ok=False, url="")

    payload = json.dumps(_structured_payload(draft)).encode("utf-8")
    request = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {settings.internal_api_key}",
            "X-PNN-Secret": settings.pnn_shared_secret,
        },
    )

    logger.info("PNN bot callback URL: %s", url)
    try:
        with urlopen(request, timeout=10) as response:
            status_code = response.getcode()
            body = response.read().decode("utf-8", errors="replace")
            if 200 <= status_code < 300:
                logger.info("PNN bot callback succeeded: status=%s body=%s", status_code, body[:500])
                return CallbackResult(
                    attempted=True,
                    ok=True,
                    url=url,
                    status_code=status_code,
                    body=body,
                )

            logger.error("PNN bot callback failed: status=%s body=%s", status_code, body[:1000])
            return CallbackResult(
                attempted=True,
                ok=False,
                url=url,
                status_code=status_code,
                body=body,
            )
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        logger.error("PNN bot callback failed: status=%s body=%s", exc.code, body[:1000])
        return CallbackResult(
            attempted=True,
            ok=False,
            url=url,
            status_code=exc.code,
            body=body,
            error=str(exc),
        )
    except URLError as exc:
        logger.error("PNN bot callback failed: url=%s error=%s", url, exc)
        return CallbackResult(attempted=True, ok=False, url=url, error=str(exc))
    except Exception as exc:
        logger.exception("PNN bot callback failed unexpectedly: url=%s", url)
        return CallbackResult(attempted=True, ok=False, url=url, error=str(exc))


def _structured_payload(draft: dict) -> dict:
    draft_id = str(draft.get("draft_id") or draft.get("id") or "")
    return {
        "draft_id": draft_id,
        "headline": draft.get("headline") or draft.get("title") or "",
        "summary": draft.get("summary") or draft.get("source_summary") or "",
        "sections": draft.get("sections") or [],
        "looking_ahead": draft.get("looking_ahead") or "",
        "status": draft.get("status") or "pending",
    }
