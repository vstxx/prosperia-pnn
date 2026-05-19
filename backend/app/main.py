from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, status

from .callbacks import CallbackResult, send_draft_callback
from .config import get_settings
from .database import (
    approve_draft,
    create_draft,
    get_draft,
    get_draft_by_public_id,
    get_drafts_for_message_ids,
    get_latest_draft,
    get_message,
    get_draft_for_message,
    get_recent_messages,
    init_db,
    insert_raw_discord_message,
    list_drafts,
    mark_published,
    reject_draft,
    rewrite_draft,
    update_draft_review_message,
)
from .drafting import (
    DraftGenerationError,
    analyze_news_window,
    article_from_draft_content,
    draft_content_from_article,
    generate_journalist_draft,
    should_create_news_draft,
)
from .models import ArticleDraftIn, DraftActionIn, MessageIn, ReviewMessageIn, RewriteDraftIn
from .scoring import ScoreResult, score_message
from .security import require_internal_api_key, require_shared_secret


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pnn.backend")
settings = get_settings()
app = FastAPI(title=settings.app_name, version="0.1.0")


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/")
def root() -> dict:
    return {
        "name": settings.app_name,
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _require_draft(draft_id: int) -> dict:
    draft = get_draft(draft_id)
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return draft


def _require_pnn_draft(draft_id: str) -> dict:
    draft = get_draft_by_public_id(draft_id)
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")
    return draft


def _ensure_pending(draft: dict) -> None:
    if draft["status"] != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Draft is {draft['status']} and cannot be changed by this action",
        )


def _content_preview(content: str) -> str:
    return " ".join((content or "").split())[:80]


def _callback_payload(callback_result: CallbackResult) -> dict:
    return {
        "attempted": callback_result.attempted,
        "ok": callback_result.ok,
        "url": callback_result.url,
        "status_code": callback_result.status_code,
        "body": callback_result.body,
        "error": callback_result.error,
    }


def _pnn_draft_payload(draft: dict) -> dict:
    return {
        "draft_id": str(draft.get("draft_id") or draft.get("id")),
        "headline": draft.get("headline") or draft.get("title") or "",
        "summary": draft.get("summary") or draft.get("source_summary") or "",
        "sections": draft.get("sections") or [],
        "looking_ahead": draft.get("looking_ahead") or "",
        "status": draft.get("status") or "pending",
        "created_at": draft.get("created_at"),
        "updated_at": draft.get("updated_at"),
        "review_message_id": draft.get("review_message_id"),
        "review_channel_id": draft.get("review_channel_id"),
        "published_message_id": draft.get("published_message_id"),
    }


def _create_ai_draft_from_recent_window() -> tuple[dict, CallbackResult, dict]:
    recent_messages = get_recent_messages(settings.analysis_window_minutes)
    if not recent_messages:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No recent livechat messages are available for draft generation",
        )

    analysis = analyze_news_window(recent_messages, settings)
    candidate_ids = analysis.fact_pack.get("candidate_message_ids") or []
    if not candidate_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No candidate source messages were found in the recent livechat window",
        )

    existing_topic_drafts = get_drafts_for_message_ids(candidate_ids)
    if existing_topic_drafts:
        existing = existing_topic_drafts[0]
        callback_result = (
            CallbackResult(
                attempted=False,
                ok=True,
                url=settings.bot_review_endpoint,
                body="Draft already has a review message",
            )
            if existing.get("review_message_id")
            else send_draft_callback(existing, settings)
        )
        return existing, callback_result, analysis.to_dict()

    should_draft, gate_reason = should_create_news_draft(analysis, settings)
    if not should_draft:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "Recent livechat data did not pass the draft gate",
                "gate_reason": gate_reason,
                "analysis": analysis.to_dict(),
            },
        )

    source_message = get_message(candidate_ids[0])
    if not source_message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft source message not found")

    draft_score = _score_from_analysis(analysis)
    try:
        draft_content = generate_journalist_draft(analysis, settings)
    except DraftGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Journalist draft generation failed: {exc.reason}",
        ) from exc

    draft = create_draft(source_message, draft_score, draft_content)
    callback_result = send_draft_callback(draft, settings)
    if not callback_result.ok:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Draft was created, but the bot review callback failed",
                "draft": _pnn_draft_payload(draft),
                "callback": _callback_payload(callback_result),
            },
        )
    return draft, callback_result, analysis.to_dict()


def _log_message_decision(
    record: dict,
    score: int,
    threshold: int,
    decision: str,
    draft_id: Optional[int] = None,
) -> None:
    logger.info(
        "PNN ingest: channel_id=%s content_preview=%r score=%s threshold=%s force_drafts=%s decision=%s draft_id=%s bot_callback_url=%s",
        record.get("channel_id"),
        _content_preview(record.get("content") or ""),
        score,
        threshold,
        settings.pnn_force_drafts,
        decision,
        draft_id,
        settings.pnn_bot_callback_url,
    )


def _score_from_analysis(analysis) -> ScoreResult:
    decision = analysis.decision
    reasons = [
        f"analyst_topic={decision.topic}",
        f"analyst_action={decision.recommended_action}",
        f"analyst_confidence={decision.confidence}",
        decision.reason,
    ]
    return ScoreResult(score=decision.newsworthiness_score, reasons=reasons)


@app.post("/api/messages", dependencies=[Depends(require_shared_secret)])
def ingest_message(message: MessageIn) -> dict:
    record = message.to_record()
    score_result = score_message(record.get("content") or "", record.get("attachments") or [])
    stored_message, created = insert_raw_discord_message(record, score_result)
    threshold = settings.news_score_threshold
    logger.info(
        "PNN newsroom: incoming message stored raw_id=%s created=%s discord_message_id=%s channel_id=%s",
        stored_message["id"],
        created,
        stored_message.get("discord_message_id"),
        stored_message.get("channel_id"),
    )

    existing_draft = get_draft_for_message(stored_message["id"])
    if existing_draft:
        if existing_draft.get("review_message_id"):
            callback_result = CallbackResult(
                attempted=False,
                ok=True,
                url=settings.pnn_bot_callback_url,
                body="Draft already has a review message",
            )
            decision = "existing_draft_already_in_review"
        else:
            callback_result = send_draft_callback(existing_draft, settings)
            decision = "existing_draft_callback"
        _log_message_decision(record, stored_message["score"], threshold, decision, existing_draft["id"])
        return {
            "stored": created,
            "message_id": stored_message["id"],
            "score": stored_message["score"],
            "reasons": stored_message["score_reasons"],
            "draft_created": False,
            "draft": existing_draft,
            "callback": _callback_payload(callback_result),
        }
    if not created:
        logger.info("PNN newsroom: ignored duplicate raw message without existing draft")
        return {
            "stored": False,
            "message_id": stored_message["id"],
            "score": stored_message["score"],
            "reasons": stored_message["score_reasons"],
            "draft_created": False,
            "draft": None,
            "callback": None,
            "analysis": None,
            "ignored_reason": "duplicate_raw_message",
        }

    recent_messages = get_recent_messages(settings.analysis_window_minutes)
    analysis = analyze_news_window(recent_messages, settings, focus_message=stored_message)
    existing_topic_drafts = get_drafts_for_message_ids(analysis.fact_pack.get("candidate_message_ids") or [])
    if existing_topic_drafts:
        reason = "existing_topic_draft_in_window"
        logger.info(
            "PNN newsroom: draft ignored reason=%s existing_draft_id=%s",
            reason,
            existing_topic_drafts[0]["id"],
        )
        _log_message_decision(record, analysis.decision.newsworthiness_score, threshold, reason)
        return {
            "stored": created,
            "message_id": stored_message["id"],
            "score": analysis.decision.newsworthiness_score,
            "reasons": [analysis.decision.reason],
            "draft_created": False,
            "draft": existing_topic_drafts[0],
            "callback": None,
            "analysis": analysis.to_dict(),
            "ignored_reason": reason,
        }

    should_draft, gate_reason = should_create_news_draft(analysis, settings)
    if not should_draft:
        logger.info(
            "PNN newsroom: draft ignored reason=%s analyst_action=%s score=%s confidence=%s",
            gate_reason,
            analysis.decision.recommended_action,
            analysis.decision.newsworthiness_score,
            analysis.decision.confidence,
        )
        _log_message_decision(record, analysis.decision.newsworthiness_score, threshold, gate_reason)
        return {
            "stored": created,
            "message_id": stored_message["id"],
            "score": analysis.decision.newsworthiness_score,
            "reasons": [analysis.decision.reason],
            "draft_created": False,
            "draft": None,
            "callback": None,
            "analysis": analysis.to_dict(),
            "ignored_reason": gate_reason,
        }

    draft_score = _score_from_analysis(analysis)
    try:
        draft_content = generate_journalist_draft(analysis, settings)
    except DraftGenerationError as exc:
        reason = f"journalist_generation_failed:{exc.reason}"
        logger.error("PNN newsroom: draft ignored reason=%s", reason)
        return {
            "stored": created,
            "message_id": stored_message["id"],
            "score": analysis.decision.newsworthiness_score,
            "reasons": [analysis.decision.reason],
            "draft_created": False,
            "draft": None,
            "callback": None,
            "analysis": analysis.to_dict(),
            "ignored_reason": reason,
        }
    draft = create_draft(stored_message, draft_score, draft_content)
    callback_result = send_draft_callback(draft, settings)
    logger.info(
        "PNN newsroom: draft created draft_id=%s topic=%r score=%s confidence=%s",
        draft["id"],
        analysis.decision.topic,
        analysis.decision.newsworthiness_score,
        analysis.decision.confidence,
    )
    _log_message_decision(record, analysis.decision.newsworthiness_score, threshold, "drafted", draft["id"])
    return {
        "stored": created,
        "message_id": stored_message["id"],
        "score": analysis.decision.newsworthiness_score,
        "reasons": draft_score.reasons,
        "draft_created": True,
        "draft": draft,
        "callback": _callback_payload(callback_result),
        "analysis": analysis.to_dict(),
    }


@app.post("/api/test-draft", dependencies=[Depends(require_shared_secret)])
def test_draft() -> dict:
    fake_id = f"test-{uuid4()}"
    record = {
        "message_id": fake_id,
        "guild_id": "test-guild",
        "channel_id": "test-channel",
        "channel_name": "pnn-test",
        "author_id": "pnn-test",
        "author_display": "PNN Test",
        "content": "Breaking test update from Prosperia News Network. This draft was created by POST /api/test-draft.",
        "jump_url": None,
        "created_at": None,
        "attachments": [],
        "minecraft_events": [],
        "staff_confirmed": False,
    }
    score_result = ScoreResult(score=100, reasons=["manual test draft"])
    stored_message, _ = insert_raw_discord_message(record, score_result)
    analysis = analyze_news_window([stored_message], settings, focus_message=stored_message)
    draft_score = _score_from_analysis(analysis)
    draft_content = generate_journalist_draft(analysis, settings)
    draft = create_draft(stored_message, draft_score, draft_content)
    callback_result = send_draft_callback(draft, settings)
    _log_message_decision(record, score_result.score, settings.news_score_threshold, "test_drafted", draft["id"])

    if not callback_result.ok:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Test draft was created, but the bot callback failed",
                "draft": draft,
                "callback": _callback_payload(callback_result),
            },
        )

    return {"draft": draft, "callback": _callback_payload(callback_result)}


@app.post("/api/analyze-now", dependencies=[Depends(require_shared_secret)])
def analyze_now() -> dict:
    recent_messages = get_recent_messages(settings.analysis_window_minutes)
    analysis = analyze_news_window(recent_messages, settings)
    should_draft, gate_reason = should_create_news_draft(analysis, settings)
    logger.info(
        "PNN newsroom: manual analyze-now decision=%s score=%s confidence=%s gate=%s",
        analysis.decision.recommended_action,
        analysis.decision.newsworthiness_score,
        analysis.decision.confidence,
        gate_reason,
    )
    return {
        "window_minutes": settings.analysis_window_minutes,
        "messages_analyzed": len(recent_messages),
        "would_create_draft": should_draft,
        "gate_reason": gate_reason,
        "analysis": analysis.to_dict(include_fact_pack=True),
    }


@app.post("/pnn/drafts", dependencies=[Depends(require_internal_api_key)])
def create_pnn_draft(payload: ArticleDraftIn) -> dict:
    article = payload.to_record()
    if article.get("draft_id"):
        existing = get_draft_by_public_id(article["draft_id"])
        if existing:
            return _pnn_draft_payload(existing)
    score_result = ScoreResult(score=100, reasons=["pnn structured draft"])
    source_record = {
        "message_id": f"pnn-draft-{article.get('draft_id') or uuid4()}",
        "guild_id": "pnn-internal",
        "channel_id": "pnn-internal",
        "channel_name": "pnn-internal",
        "author_id": "pnn-backend",
        "author_display": "Prosperia News Network",
        "content": article.get("summary") or article.get("headline") or "",
        "jump_url": None,
        "created_at": None,
        "attachments": [],
        "minecraft_events": [],
        "staff_confirmed": True,
    }
    stored_message, _ = insert_raw_discord_message(source_record, score_result)
    draft_content = draft_content_from_article(article)
    draft = create_draft(
        stored_message,
        score_result,
        draft_content,
        public_draft_id=article.get("draft_id") or None,
    )
    callback_result = send_draft_callback(draft, settings)
    if not callback_result.ok:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail={
                "message": "Draft was created, but the bot review callback failed",
                "draft": _pnn_draft_payload(draft),
                "callback": _callback_payload(callback_result),
            },
        )
    return _pnn_draft_payload(draft)


@app.post("/pnn/generate-draft", dependencies=[Depends(require_internal_api_key)])
def generate_pnn_draft() -> dict:
    draft, callback_result, analysis = _create_ai_draft_from_recent_window()
    return {
        "draft": _pnn_draft_payload(draft),
        "callback": _callback_payload(callback_result),
        "analysis": analysis,
    }


@app.get("/pnn/drafts/latest", dependencies=[Depends(require_internal_api_key)])
def get_latest_pnn_draft(status_filter: Optional[str] = Query(default="pending", alias="status")) -> dict:
    draft = get_latest_draft(status=status_filter)
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No matching draft found")
    return _pnn_draft_payload(draft)


@app.get("/pnn/drafts/{draft_id}", dependencies=[Depends(require_internal_api_key)])
def get_pnn_draft(draft_id: str) -> dict:
    return _pnn_draft_payload(_require_pnn_draft(draft_id))


@app.post("/pnn/drafts/{draft_id}/review-message", dependencies=[Depends(require_internal_api_key)])
def save_pnn_review_message(draft_id: str, payload: ReviewMessageIn) -> dict:
    draft = _require_pnn_draft(draft_id)
    updated = update_draft_review_message(draft["id"], payload.channel_id, payload.message_id)
    return _pnn_draft_payload(updated)


@app.post("/pnn/drafts/{draft_id}/publish", dependencies=[Depends(require_internal_api_key)])
def publish_pnn_draft(draft_id: str, payload: DraftActionIn) -> dict:
    draft = _require_pnn_draft(draft_id)
    _ensure_pending(draft)
    updated = mark_published(
        draft["id"],
        reviewer_id=payload.reviewer_id,
        published_message_id=payload.published_message_id,
    )
    return _pnn_draft_payload(updated)


@app.post("/pnn/drafts/{draft_id}/reject", dependencies=[Depends(require_internal_api_key)])
def reject_pnn_draft(draft_id: str, payload: DraftActionIn) -> dict:
    draft = _require_pnn_draft(draft_id)
    _ensure_pending(draft)
    rejected = reject_draft(draft["id"], reviewer_id=payload.reviewer_id)
    return _pnn_draft_payload(rejected)


@app.post("/pnn/drafts/{draft_id}/rewrite", dependencies=[Depends(require_internal_api_key)])
def rewrite_pnn_draft(draft_id: str, payload: RewriteDraftIn) -> dict:
    draft = _require_pnn_draft(draft_id)
    _ensure_pending(draft)

    message = get_message(draft["message_id"])
    if not message:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft source message not found")
    recent_messages = get_recent_messages(settings.analysis_window_minutes)
    if all(item.get("id") != message.get("id") for item in recent_messages):
        recent_messages.insert(0, message)
    analysis = analyze_news_window(recent_messages, settings, focus_message=message)
    regenerated = generate_journalist_draft(analysis, settings)
    article = article_from_draft_content(regenerated).to_dict()
    updated = rewrite_draft(
        draft["id"],
        title=regenerated.title,
        body=regenerated.body,
        reviewer_id=payload.reviewer_id,
        article=article,
    )
    return _pnn_draft_payload(updated)


@app.get("/api/drafts", dependencies=[Depends(require_shared_secret)])
def drafts(
    status_filter: Optional[str] = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=200),
) -> dict:
    return {"drafts": list_drafts(status=status_filter, limit=limit)}


@app.get("/api/drafts/{draft_id}", dependencies=[Depends(require_shared_secret)])
def draft_detail(draft_id: int) -> dict:
    return {"draft": _require_draft(draft_id)}


@app.patch("/api/drafts/{draft_id}/review-message", dependencies=[Depends(require_shared_secret)])
def save_review_message(draft_id: int, payload: ReviewMessageIn) -> dict:
    _require_draft(draft_id)
    draft = update_draft_review_message(draft_id, payload.channel_id, payload.message_id)
    return {"draft": draft}


@app.post("/api/drafts/{draft_id}/rewrite", dependencies=[Depends(require_shared_secret)])
def rewrite(draft_id: int, payload: RewriteDraftIn) -> dict:
    draft = _require_draft(draft_id)
    _ensure_pending(draft)

    title = (payload.title or "").strip()
    body = (payload.body or "").strip()
    if not title or not body:
        message = get_message(draft["message_id"])
        if not message:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft source message not found")
        recent_messages = get_recent_messages(settings.analysis_window_minutes)
        if all(item.get("id") != message.get("id") for item in recent_messages):
            recent_messages.insert(0, message)
        analysis = analyze_news_window(recent_messages, settings, focus_message=message)
        regenerated = generate_journalist_draft(analysis, settings)
        title = title or regenerated.title
        body = body or regenerated.body
        article = article_from_draft_content(regenerated).to_dict()
    else:
        article = None

    updated = rewrite_draft(draft_id, title=title, body=body, reviewer_id=payload.reviewer_id, article=article)
    return {"draft": updated}


@app.post("/api/drafts/{draft_id}/publish", dependencies=[Depends(require_shared_secret)])
def publish(draft_id: int, payload: DraftActionIn) -> dict:
    draft = _require_draft(draft_id)
    _ensure_pending(draft)
    approved = approve_draft(draft_id, reviewer_id=payload.reviewer_id)
    return {"draft": approved}


@app.post("/api/drafts/{draft_id}/reject", dependencies=[Depends(require_shared_secret)])
def reject(draft_id: int, payload: DraftActionIn) -> dict:
    draft = _require_draft(draft_id)
    _ensure_pending(draft)
    rejected = reject_draft(draft_id, reviewer_id=payload.reviewer_id)
    return {"draft": rejected}


@app.post("/api/drafts/{draft_id}/published", dependencies=[Depends(require_shared_secret)])
def published(draft_id: int, payload: DraftActionIn) -> dict:
    draft = _require_draft(draft_id)
    if draft["status"] != "approved":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Draft is {draft['status']} and cannot be marked published",
        )
    updated = mark_published(draft_id, reviewer_id=payload.reviewer_id)
    return {"draft": updated}
