from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class AttachmentIn(BaseModel):
    url: str
    name: Optional[str] = None
    content_type: Optional[str] = None


class MessageIn(BaseModel):
    message_id: str = Field(min_length=1)
    guild_id: str = Field(min_length=1)
    channel_id: str = Field(min_length=1)
    channel_name: Optional[str] = None
    author_id: str = Field(min_length=1)
    author_display: Optional[str] = None
    content: str = ""
    jump_url: Optional[str] = None
    created_at: Optional[str] = None
    attachments: list[AttachmentIn] = Field(default_factory=list)
    minecraft_events: list[dict[str, Any]] = Field(default_factory=list)
    staff_confirmed: bool = False

    def to_record(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class ReviewMessageIn(BaseModel):
    channel_id: str = Field(min_length=1)
    message_id: str = Field(min_length=1)


class DraftActionIn(BaseModel):
    reviewer_id: Optional[str] = None
    published_message_id: Optional[str] = None


class RewriteDraftIn(BaseModel):
    reviewer_id: Optional[str] = None
    title: Optional[str] = None
    body: Optional[str] = None


class ArticleSectionIn(BaseModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class ArticleDraftIn(BaseModel):
    draft_id: Optional[str] = None
    headline: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    sections: list[ArticleSectionIn] = Field(default_factory=list)
    looking_ahead: str = ""
    status: str = "pending"

    def to_record(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()
