from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from .config import get_settings
from .drafting import DraftContent, article_from_draft_content
from .scoring import ScoreResult


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    db_path = Path(get_settings().database_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


@contextmanager
def _db():
    connection = _connect()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _ensure_column(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    if column_name not in _table_columns(connection, table_name):
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {definition}")


def init_db() -> None:
    with _db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                discord_message_id TEXT NOT NULL UNIQUE,
                guild_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                channel_name TEXT,
                author_id TEXT NOT NULL,
                author_display TEXT,
                content TEXT NOT NULL,
                jump_url TEXT,
                attachments_json TEXT NOT NULL DEFAULT '[]',
                minecraft_events_json TEXT NOT NULL DEFAULT '[]',
                staff_confirmed INTEGER NOT NULL DEFAULT 0,
                score INTEGER NOT NULL,
                score_reasons_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT,
                received_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS drafts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draft_id TEXT UNIQUE,
                message_id INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                headline TEXT,
                summary TEXT,
                sections_json TEXT NOT NULL DEFAULT '[]',
                looking_ahead TEXT,
                source_summary TEXT NOT NULL,
                score INTEGER NOT NULL,
                rationale_json TEXT NOT NULL DEFAULT '[]',
                review_channel_id TEXT,
                review_message_id TEXT,
                published_message_id TEXT,
                reviewer_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                approved_at TEXT,
                rejected_at TEXT,
                published_at TEXT,
                FOREIGN KEY (message_id) REFERENCES messages(id)
            );

            CREATE INDEX IF NOT EXISTS idx_messages_channel ON messages(channel_id);
            CREATE INDEX IF NOT EXISTS idx_messages_received_at ON messages(received_at);
            CREATE INDEX IF NOT EXISTS idx_drafts_status ON drafts(status);
            CREATE INDEX IF NOT EXISTS idx_drafts_message ON drafts(message_id);

            CREATE VIEW IF NOT EXISTS raw_discord_messages AS
            SELECT * FROM messages;

            CREATE VIEW IF NOT EXISTS RawDiscordMessage AS
            SELECT * FROM messages;
            """
        )
        _ensure_column(
            connection,
            "messages",
            "minecraft_events_json",
            "minecraft_events_json TEXT NOT NULL DEFAULT '[]'",
        )
        _ensure_column(
            connection,
            "messages",
            "staff_confirmed",
            "staff_confirmed INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(connection, "drafts", "draft_id", "draft_id TEXT")
        _ensure_column(connection, "drafts", "headline", "headline TEXT")
        _ensure_column(connection, "drafts", "summary", "summary TEXT")
        _ensure_column(connection, "drafts", "sections_json", "sections_json TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(connection, "drafts", "looking_ahead", "looking_ahead TEXT")
        _ensure_column(connection, "drafts", "published_message_id", "published_message_id TEXT")
        connection.execute("UPDATE drafts SET draft_id = CAST(id AS TEXT) WHERE draft_id IS NULL OR draft_id = ''")
        connection.execute("UPDATE drafts SET headline = title WHERE headline IS NULL OR headline = ''")
        connection.execute("UPDATE drafts SET summary = source_summary WHERE summary IS NULL OR summary = ''")
        connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_drafts_draft_id ON drafts(draft_id)")


def _json_loads(value: Optional[str], fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return fallback


def _message_from_row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    data["attachments"] = _json_loads(data.pop("attachments_json", "[]"), [])
    data["minecraft_events"] = _json_loads(data.pop("minecraft_events_json", "[]"), [])
    data["staff_confirmed"] = bool(data.get("staff_confirmed", 0))
    data["score_reasons"] = _json_loads(data.pop("score_reasons_json", "[]"), [])
    return data


def _draft_from_row(row: Optional[sqlite3.Row]) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    data["rationale"] = _json_loads(data.pop("rationale_json", "[]"), [])
    data["sections"] = _json_loads(data.pop("sections_json", "[]"), [])
    data["draft_id"] = str(data.get("draft_id") or data.get("id"))
    data["headline"] = data.get("headline") or data.get("title") or ""
    data["summary"] = data.get("summary") or data.get("source_summary") or ""
    data["looking_ahead"] = data.get("looking_ahead") or ""
    data["article"] = {
        "draft_id": data["draft_id"],
        "headline": data["headline"],
        "summary": data["summary"],
        "sections": data["sections"],
        "looking_ahead": data["looking_ahead"],
        "status": data.get("status") or "pending",
    }

    source_keys = {
        "source_discord_message_id",
        "source_guild_id",
        "source_channel_id",
        "source_channel_name",
        "source_author_id",
        "source_author_display",
        "source_content",
        "source_jump_url",
        "source_minecraft_events_json",
        "source_staff_confirmed",
        "source_created_at",
    }
    if source_keys.intersection(data):
        source_minecraft_events = _json_loads(data.pop("source_minecraft_events_json", "[]"), [])
        source_staff_confirmed = bool(data.pop("source_staff_confirmed", 0))
        data["source"] = {
            "discord_message_id": data.pop("source_discord_message_id", None),
            "guild_id": data.pop("source_guild_id", None),
            "channel_id": data.pop("source_channel_id", None),
            "channel_name": data.pop("source_channel_name", None),
            "author_id": data.pop("source_author_id", None),
            "author_display": data.pop("source_author_display", None),
            "content": data.pop("source_content", None),
            "jump_url": data.pop("source_jump_url", None),
            "minecraft_events": source_minecraft_events,
            "staff_confirmed": source_staff_confirmed,
            "created_at": data.pop("source_created_at", None),
        }
    return data


def insert_raw_discord_message(message: dict[str, Any], score_result: ScoreResult) -> tuple[dict[str, Any], bool]:
    now = utc_now()
    attachments = message.get("attachments") or []
    minecraft_events = message.get("minecraft_events") or []
    with _db() as connection:
        try:
            cursor = connection.execute(
                """
                INSERT INTO messages (
                    discord_message_id, guild_id, channel_id, channel_name, author_id,
                    author_display, content, jump_url, attachments_json, minecraft_events_json,
                    staff_confirmed, score,
                    score_reasons_json, created_at, received_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message["message_id"],
                    message["guild_id"],
                    message["channel_id"],
                    message.get("channel_name"),
                    message["author_id"],
                    message.get("author_display"),
                    message.get("content") or "",
                    message.get("jump_url"),
                    json.dumps(attachments),
                    json.dumps(minecraft_events),
                    1 if message.get("staff_confirmed") else 0,
                    score_result.score,
                    json.dumps(score_result.reasons),
                    message.get("created_at"),
                    now,
                ),
            )
            row = connection.execute("SELECT * FROM messages WHERE id = ?", (cursor.lastrowid,)).fetchone()
            return _message_from_row(row), True
        except sqlite3.IntegrityError:
            row = connection.execute(
                "SELECT * FROM messages WHERE discord_message_id = ?",
                (message["message_id"],),
            ).fetchone()
            return _message_from_row(row), False


def insert_message(message: dict[str, Any], score_result: ScoreResult) -> tuple[dict[str, Any], bool]:
    return insert_raw_discord_message(message, score_result)


def get_message(message_id: int) -> Optional[dict[str, Any]]:
    with _db() as connection:
        row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        return _message_from_row(row)


def get_recent_messages(window_minutes: int, limit: int = 250) -> list[dict[str, Any]]:
    window_minutes = max(1, min(window_minutes, 60))
    limit = max(1, min(limit, 500))
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=window_minutes)).isoformat()

    with _db() as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM messages
            WHERE received_at >= ?
            ORDER BY received_at DESC, id DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
        return [_message_from_row(row) for row in rows]


def get_recent_channel_messages(
    channel_id: str,
    before_message_id: Optional[int] = None,
    limit: int = 25,
) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 50))
    params: list[Any] = [channel_id]
    where = "WHERE channel_id = ?"
    if before_message_id is not None:
        where += " AND id < ?"
        params.append(before_message_id)
    params.append(limit)

    with _db() as connection:
        rows = connection.execute(
            f"""
            SELECT *
            FROM messages
            {where}
            ORDER BY id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_message_from_row(row) for row in rows]


def get_drafts_for_message_ids(message_ids: list[int]) -> list[dict[str, Any]]:
    if not message_ids:
        return []
    unique_ids = sorted(set(message_ids))
    placeholders = ",".join("?" for _ in unique_ids)
    with _db() as connection:
        rows = connection.execute(
            f"""
            SELECT d.*, m.discord_message_id AS source_discord_message_id,
                   m.guild_id AS source_guild_id,
                   m.channel_id AS source_channel_id,
                   m.channel_name AS source_channel_name,
                   m.author_id AS source_author_id,
                   m.author_display AS source_author_display,
                   m.content AS source_content,
                   m.jump_url AS source_jump_url,
                   m.minecraft_events_json AS source_minecraft_events_json,
                   m.staff_confirmed AS source_staff_confirmed,
                   m.created_at AS source_created_at
            FROM drafts d
            JOIN messages m ON m.id = d.message_id
            WHERE d.message_id IN ({placeholders})
            ORDER BY d.created_at DESC
            """,
            unique_ids,
        ).fetchall()
        return [_draft_from_row(row) for row in rows]


def get_draft_for_message(message_id: int) -> Optional[dict[str, Any]]:
    with _db() as connection:
        row = connection.execute(
            """
            SELECT d.*, m.discord_message_id AS source_discord_message_id,
                   m.guild_id AS source_guild_id,
                   m.channel_id AS source_channel_id,
                   m.channel_name AS source_channel_name,
                   m.author_id AS source_author_id,
                   m.author_display AS source_author_display,
                   m.content AS source_content,
                   m.jump_url AS source_jump_url,
                   m.minecraft_events_json AS source_minecraft_events_json,
                   m.staff_confirmed AS source_staff_confirmed,
                   m.created_at AS source_created_at
            FROM drafts d
            JOIN messages m ON m.id = d.message_id
            WHERE d.message_id = ?
            ORDER BY d.id DESC
            LIMIT 1
            """,
            (message_id,),
        ).fetchone()
        return _draft_from_row(row)


def create_draft(
    message: dict[str, Any],
    score_result: ScoreResult,
    draft: DraftContent,
    public_draft_id: Optional[str] = None,
) -> dict[str, Any]:
    now = utc_now()
    article = article_from_draft_content(draft)
    public_draft_id = public_draft_id or str(uuid4())
    with _db() as connection:
        cursor = connection.execute(
            """
            INSERT INTO drafts (
                draft_id, message_id, status, title, body, headline, summary, sections_json,
                looking_ahead, source_summary, score, rationale_json,
                created_at, updated_at
            )
            VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                public_draft_id,
                message["id"],
                draft.title,
                draft.body,
                article.headline,
                article.summary,
                json.dumps([{"title": section.title, "content": section.content} for section in article.sections]),
                article.looking_ahead,
                draft.source_summary,
                score_result.score,
                json.dumps(score_result.reasons),
                now,
                now,
            ),
        )
    return get_draft(cursor.lastrowid)


def get_draft(draft_id: int) -> Optional[dict[str, Any]]:
    with _db() as connection:
        row = connection.execute(
            """
            SELECT d.*, m.discord_message_id AS source_discord_message_id,
                   m.guild_id AS source_guild_id,
                   m.channel_id AS source_channel_id,
                   m.channel_name AS source_channel_name,
                   m.author_id AS source_author_id,
                   m.author_display AS source_author_display,
                   m.content AS source_content,
                   m.jump_url AS source_jump_url,
                   m.minecraft_events_json AS source_minecraft_events_json,
                   m.staff_confirmed AS source_staff_confirmed,
                   m.created_at AS source_created_at
            FROM drafts d
            JOIN messages m ON m.id = d.message_id
            WHERE d.id = ?
            """,
            (draft_id,),
        ).fetchone()
        return _draft_from_row(row)


def get_draft_by_public_id(draft_id: str) -> Optional[dict[str, Any]]:
    with _db() as connection:
        row = connection.execute(
            """
            SELECT d.*, m.discord_message_id AS source_discord_message_id,
                   m.guild_id AS source_guild_id,
                   m.channel_id AS source_channel_id,
                   m.channel_name AS source_channel_name,
                   m.author_id AS source_author_id,
                   m.author_display AS source_author_display,
                   m.content AS source_content,
                   m.jump_url AS source_jump_url,
                   m.minecraft_events_json AS source_minecraft_events_json,
                   m.staff_confirmed AS source_staff_confirmed,
                   m.created_at AS source_created_at
            FROM drafts d
            JOIN messages m ON m.id = d.message_id
            WHERE d.draft_id = ? OR CAST(d.id AS TEXT) = ?
            """,
            (draft_id, draft_id),
        ).fetchone()
        return _draft_from_row(row)


def list_drafts(status: Optional[str] = None, limit: int = 50) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 200))
    params: list[Any] = []
    where = ""
    if status:
        where = "WHERE d.status = ?"
        params.append(status)
    params.append(limit)

    with _db() as connection:
        rows = connection.execute(
            f"""
            SELECT d.*, m.discord_message_id AS source_discord_message_id,
                   m.guild_id AS source_guild_id,
                   m.channel_id AS source_channel_id,
                   m.channel_name AS source_channel_name,
                   m.author_id AS source_author_id,
                   m.author_display AS source_author_display,
                   m.content AS source_content,
                   m.jump_url AS source_jump_url,
                   m.minecraft_events_json AS source_minecraft_events_json,
                   m.staff_confirmed AS source_staff_confirmed,
                   m.created_at AS source_created_at
            FROM drafts d
            JOIN messages m ON m.id = d.message_id
            {where}
            ORDER BY d.created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [_draft_from_row(row) for row in rows]


def get_latest_draft(status: Optional[str] = None) -> Optional[dict[str, Any]]:
    drafts = list_drafts(status=status, limit=1)
    if not drafts:
        return None
    return drafts[0]


def update_draft_review_message(draft_id: int, channel_id: str, message_id: str) -> Optional[dict[str, Any]]:
    now = utc_now()
    with _db() as connection:
        connection.execute(
            """
            UPDATE drafts
            SET review_channel_id = ?, review_message_id = ?, updated_at = ?
            WHERE id = ?
            """,
            (channel_id, message_id, now, draft_id),
        )
    return get_draft(draft_id)


def rewrite_draft(
    draft_id: int,
    title: str,
    body: str,
    reviewer_id: Optional[str],
    article: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    now = utc_now()
    headline = article.get("headline") if article else title
    summary = article.get("summary") if article else None
    sections = article.get("sections") if article else None
    looking_ahead = article.get("looking_ahead") if article else None
    with _db() as connection:
        connection.execute(
            """
            UPDATE drafts
            SET title = ?,
                body = ?,
                headline = COALESCE(?, headline),
                summary = COALESCE(?, summary),
                sections_json = COALESCE(?, sections_json),
                looking_ahead = COALESCE(?, looking_ahead),
                reviewer_id = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                title,
                body,
                headline,
                summary,
                json.dumps(sections) if sections is not None else None,
                looking_ahead,
                reviewer_id,
                now,
                draft_id,
            ),
        )
    return get_draft(draft_id)


def approve_draft(draft_id: int, reviewer_id: Optional[str]) -> Optional[dict[str, Any]]:
    now = utc_now()
    with _db() as connection:
        connection.execute(
            """
            UPDATE drafts
            SET status = 'approved', reviewer_id = ?, approved_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (reviewer_id, now, now, draft_id),
        )
    return get_draft(draft_id)


def reject_draft(draft_id: int, reviewer_id: Optional[str]) -> Optional[dict[str, Any]]:
    now = utc_now()
    with _db() as connection:
        connection.execute(
            """
            UPDATE drafts
            SET status = 'rejected', reviewer_id = ?, rejected_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (reviewer_id, now, now, draft_id),
        )
    return get_draft(draft_id)


def mark_published(
    draft_id: int,
    reviewer_id: Optional[str],
    published_message_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    now = utc_now()
    with _db() as connection:
        connection.execute(
            """
            UPDATE drafts
            SET status = 'published',
                reviewer_id = COALESCE(?, reviewer_id),
                published_message_id = COALESCE(?, published_message_id),
                published_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (reviewer_id, published_message_id, now, now, draft_id),
        )
    return get_draft(draft_id)
