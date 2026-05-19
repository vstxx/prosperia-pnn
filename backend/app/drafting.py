from __future__ import annotations

import json
import logging
import re
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from textwrap import shorten
from typing import Any, Optional

from .config import Settings
from .scoring import ScoreResult


logger = logging.getLogger("pnn.backend.drafting")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "https://prosperiamc.com"
OPENROUTER_TITLE = "Prosperia News Network"
MAX_TRIGGER_CONTENT_CHARS = 1200
MAX_RELATED_CONTENT_CHARS = 500
MAX_RELATED_MESSAGES = 8
MAX_ATTACHMENTS = 5
MAX_MINECRAFT_EVENTS = 10
MAX_FACT_PACK_CHARS = 9000
MAX_DRAFT_BODY_CHARS = 3900

SYSTEM_PROMPT = """You are Prosperia News Network, a serious geopolitical Minecraft news analyst.

Use only the facts provided in the fact pack.
Do not invent locations, motives, alliances, casualties, wars, betrayals, or outcomes.
If something comes only from player chat, label it as unconfirmed.
Never accuse players of cheating, scamming, exploiting, or rule-breaking unless staff_confirmed=true.
Write in a dramatic but credible wartime news style.
Output valid JSON only."""

EXPECTED_JSON_SHAPE = {
    "headline": "...",
    "body": "...",
    "confidence": "0-100",
    "risk_level": "low|medium|high",
    "recommended_action": "review|ignore",
    "uncertainties": [],
}

ANALYST_SYSTEM_PROMPT = """You are the Prosperia News Network AI Analyst.

Use only the facts provided in the fact pack.
Treat player chat as unconfirmed unless staff_confirmed=true.
Do not invent nation names, player names, motives, alliances, casualties, wars, betrayals, or outcomes.
Never accuse players of cheating, scamming, exploiting, or rule-breaking unless staff_confirmed=true.
A single random Discord message should usually be ignored or monitored, not drafted, unless it is clearly breaking news.
Output valid JSON only."""

ANALYST_JSON_SHAPE = {
    "newsworthy": True,
    "newsworthiness_score": "0-100",
    "topic": "...",
    "reason": "...",
    "recommended_action": "ignore|monitor|draft",
    "confidence": "0-100",
    "facts": ["..."],
    "uncertainties": ["..."],
}

JOURNALIST_SYSTEM_PROMPT = """You are Prosperia News Network, a serious geopolitical Minecraft news journalist.

Use only the verified analyst fact pack provided.
Do not copy the original Discord messages as the main content.
Write like a serious geopolitical newsroom covering statecraft, conflict, borders, markets, and public security.
Use concrete, restrained, newsroom language. Avoid generic AI filler such as "the situation is developing" unless the facts require it.
If something is from player chat, mark it as unconfirmed.
Do not invent nation names, player names, motives, alliances, casualties, wars, betrayals, or outcomes.
Never accuse players of cheating, scamming, exploiting, or rule-breaking unless staff_confirmed=true.
Output valid JSON only."""

JOURNALIST_JSON_SHAPE = {
    "headline": "...",
    "summary": "...",
    "sections": [{"title": "...", "content": "..."}],
    "looking_ahead": "...",
    "confidence": "0-100",
    "risk_level": "low|medium|high",
    "source_note": "...",
    "uncertainties": [],
}

STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "because",
    "been",
    "being",
    "before",
    "between",
    "channel",
    "could",
    "from",
    "have",
    "into",
    "just",
    "like",
    "message",
    "minecraft",
    "more",
    "only",
    "over",
    "player",
    "players",
    "said",
    "same",
    "some",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "with",
    "would",
    "your",
}

TOPIC_KEYWORDS = {
    "alliance",
    "army",
    "auction",
    "bank",
    "battle",
    "border",
    "capture",
    "captured",
    "claim",
    "claims",
    "conflict",
    "coup",
    "crisis",
    "diplomacy",
    "economy",
    "embargo",
    "front",
    "market",
    "nation",
    "nations",
    "peace",
    "raid",
    "raided",
    "relic",
    "siege",
    "stockpile",
    "trade",
    "treaty",
    "war",
}

BREAKING_KEYWORDS = {
    "breaking",
    "captured",
    "capture",
    "declared war",
    "invasion",
    "raid",
    "raided",
    "relic",
    "siege",
    "surrender",
    "war",
}

RULE_BREAKING_PATTERN = re.compile(
    r"\b(cheat(?:ing|ed|er|ers|s)?|scam(?:ming|med|mer|mers|s)?|"
    r"exploit(?:ing|ed|er|ers|s)?|rule[- ]?breaking|hack(?:ing|ed|er|ers|s)?|"
    r"x-?ray(?:ing|ed)?|dupe(?:d|s|r|rs)?|duping)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ArticleSection:
    title: str
    content: str


@dataclass(frozen=True)
class ArticleContent:
    headline: str
    summary: str
    sections: list[ArticleSection]
    looking_ahead: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "headline": self.headline,
            "summary": self.summary,
            "sections": [asdict(section) for section in self.sections],
            "looking_ahead": self.looking_ahead,
        }


@dataclass(frozen=True)
class DraftContent:
    title: str
    body: str
    source_summary: str
    article: Optional[ArticleContent] = None


@dataclass(frozen=True)
class AnalystDecision:
    newsworthy: bool
    newsworthiness_score: int
    topic: str
    reason: str
    recommended_action: str
    confidence: int
    facts: list[str]
    uncertainties: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NewsroomAnalysis:
    decision: AnalystDecision
    fact_pack: dict[str, Any]
    verified_fact_pack: dict[str, Any]
    related_messages: list[dict[str, Any]]
    topic_keywords: list[str]
    prompt_token_estimate: Optional[int] = None
    analyst_used_ai: bool = False
    fallback_reason: Optional[str] = None

    def to_dict(self, include_fact_pack: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision": self.decision.to_dict(),
            "verified_fact_pack": self.verified_fact_pack,
            "related_message_count": len(self.related_messages),
            "topic_keywords": self.topic_keywords,
            "prompt_token_estimate": self.prompt_token_estimate,
            "analyst_used_ai": self.analyst_used_ai,
            "fallback_reason": self.fallback_reason,
        }
        if include_fact_pack:
            payload["fact_pack"] = self.fact_pack
        return payload


class DraftGenerationError(Exception):
    def __init__(self, reason: str, prompt_token_estimate: Optional[int] = None):
        super().__init__(reason)
        self.reason = reason
        self.prompt_token_estimate = prompt_token_estimate


def _clean_text(value: Optional[Any]) -> str:
    return " ".join(str(value or "").split())


def _clean_multiline(value: Optional[Any]) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = [" ".join(line.split()) for line in text.split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        if line:
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append("")
            blank = True
    return "\n".join(cleaned).strip()


def _clip_text(value: Optional[Any], max_chars: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)].rstrip()}..."


def _clip_multiline(value: Optional[Any], max_chars: int) -> str:
    text = _clean_multiline(value)
    if len(text) <= max_chars:
        return text
    return f"{text[: max(0, max_chars - 3)].rstrip()}..."


def _jsonish(value: Any, max_chars: int = 500) -> str:
    if value is None or isinstance(value, (str, int, float, bool)):
        return _clip_text(value, max_chars)
    try:
        raw = json.dumps(value, ensure_ascii=True, default=str, sort_keys=True)
    except TypeError:
        raw = str(value)
    return _clip_text(raw, max_chars)


def _discord_message_id(message: dict[str, Any]) -> Optional[str]:
    return message.get("discord_message_id") or message.get("message_id")


def _extract_keywords(content: Optional[str]) -> set[str]:
    words = re.findall(r"[a-z0-9_'-]{3,}", (content or "").lower())
    keywords = {word.strip("'_-") for word in words}
    return {word for word in keywords if word and word not in STOPWORDS}


def _extract_topic_keywords(content: Optional[str]) -> set[str]:
    text = content or ""
    lowered = text.lower()
    keywords = {keyword for keyword in TOPIC_KEYWORDS if re.search(rf"\b{re.escape(keyword)}\b", lowered)}
    proper_terms = {
        match.group(0).lower()
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{3,}\b", text)
        if match.group(0).lower() not in STOPWORDS
    }
    return keywords.union(proper_terms)


def _message_topic_keywords(message: dict[str, Any]) -> set[str]:
    keywords = _extract_topic_keywords(message.get("content"))
    for event in message.get("minecraft_events") or []:
        keywords.update(_extract_topic_keywords(_jsonish(event, 1000)))
    return keywords


def _has_breaking_signal(message: dict[str, Any]) -> bool:
    text = _clean_text(message.get("content")).lower()
    if any(keyword in text for keyword in BREAKING_KEYWORDS):
        return True
    for event in message.get("minecraft_events") or []:
        event_text = _jsonish(event, 1000).lower()
        if any(keyword in event_text for keyword in BREAKING_KEYWORDS):
            return True
    return bool(message.get("staff_confirmed") and (message.get("minecraft_events") or _extract_topic_keywords(text)))


def _is_related_to_trigger(candidate: dict[str, Any], trigger_keywords: set[str]) -> bool:
    if not trigger_keywords:
        return True
    return bool(_extract_keywords(candidate.get("content")).intersection(trigger_keywords))


def _attachment_facts(attachments: list[Any]) -> list[dict[str, str]]:
    facts: list[dict[str, str]] = []
    for attachment in (attachments or [])[:MAX_ATTACHMENTS]:
        if isinstance(attachment, dict):
            facts.append(
                {
                    "name": _clip_text(attachment.get("name"), 120),
                    "content_type": _clip_text(attachment.get("content_type"), 120),
                    "url": _clip_text(attachment.get("url"), 500),
                }
            )
        else:
            facts.append({"details": _jsonish(attachment, 500)})
    return facts


def _message_fact(message: dict[str, Any], max_content_chars: int) -> dict[str, Any]:
    content = _clean_text(message.get("content"))
    return {
        "source_type": "player_chat",
        "message_id": _discord_message_id(message),
        "channel_id": message.get("channel_id"),
        "channel_name": message.get("channel_name"),
        "author_id": message.get("author_id"),
        "author_display": message.get("author_display"),
        "created_at": message.get("created_at"),
        "content": _clip_text(content, max_content_chars),
        "content_truncated": len(content) > max_content_chars,
        "attachments": _attachment_facts(message.get("attachments") or []),
        "source_url": message.get("jump_url"),
        "staff_confirmed": bool(message.get("staff_confirmed")),
    }


def _source_link(role: str, message: dict[str, Any]) -> Optional[dict[str, str]]:
    url = message.get("jump_url")
    if not url:
        return None
    return {
        "role": role,
        "message_id": _discord_message_id(message) or "",
        "url": _clip_text(url, 500),
    }


def _minecraft_events_for(message: dict[str, Any], source_role: str) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    source_message_id = _discord_message_id(message) or ""
    for event in (message.get("minecraft_events") or [])[:MAX_MINECRAFT_EVENTS]:
        if isinstance(event, dict):
            normalized: dict[str, str] = {
                "source_role": source_role,
                "source_message_id": source_message_id,
            }
            for key, value in list(event.items())[:20]:
                normalized[_clip_text(key, 80)] = _jsonish(value, 500)
            events.append(normalized)
        else:
            events.append(
                {
                    "source_role": source_role,
                    "source_message_id": source_message_id,
                    "details": _jsonish(event, 800),
                }
            )
    return events


def _fact_pack_size(fact_pack: dict[str, Any]) -> int:
    return len(json.dumps(fact_pack, ensure_ascii=True, separators=(",", ":")))


def _fit_fact_pack(fact_pack: dict[str, Any]) -> dict[str, Any]:
    while _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS and fact_pack["recent_related_messages"]:
        fact_pack["recent_related_messages"].pop()
    while _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS and fact_pack["minecraft_events"]:
        fact_pack["minecraft_events"].pop()
    if _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS:
        trigger = fact_pack["triggering_message"]
        trigger["content"] = _clip_text(trigger.get("content"), 700)
        trigger["content_truncated"] = True
    return fact_pack


def build_fact_pack(
    message: dict[str, Any],
    score_result: ScoreResult,
    related_messages: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    trigger_keywords = _extract_keywords(message.get("content"))
    trigger = _message_fact(message, MAX_TRIGGER_CONTENT_CHARS)
    recent_related: list[dict[str, Any]] = []
    source_links: list[dict[str, str]] = []
    minecraft_events = _minecraft_events_for(message, "triggering_message")

    trigger_link = _source_link("triggering_message", message)
    if trigger_link:
        source_links.append(trigger_link)

    for candidate in related_messages or []:
        if len(recent_related) >= MAX_RELATED_MESSAGES:
            break
        if candidate.get("id") == message.get("id"):
            continue
        if candidate.get("channel_id") != message.get("channel_id"):
            continue
        if not _is_related_to_trigger(candidate, trigger_keywords):
            continue

        recent_related.append(_message_fact(candidate, MAX_RELATED_CONTENT_CHARS))
        link = _source_link("recent_related_message", candidate)
        if link:
            source_links.append(link)
        minecraft_events.extend(_minecraft_events_for(candidate, "recent_related_message"))

    fact_pack = {
        "schema": "pnn_fact_pack_v1",
        "staff_confirmed": bool(message.get("staff_confirmed")),
        "rules": {
            "use_only_provided_facts": True,
            "player_chat_is_unconfirmed": True,
            "no_rule_breaking_accusations_without_staff_confirmed": True,
        },
        "triggering_message": trigger,
        "recent_related_messages": recent_related,
        "minecraft_events": minecraft_events[:MAX_MINECRAFT_EVENTS],
        "source_message_links": source_links[: MAX_RELATED_MESSAGES + 1],
        "backend_signal": {
            "score": score_result.score,
            "reasons": [_clip_text(reason, 180) for reason in score_result.reasons],
        },
    }
    return _fit_fact_pack(fact_pack)


def _build_user_prompt(fact_pack: dict[str, Any], variant: str) -> str:
    prompt_goal = "Rewrite the existing draft material" if variant == "rewrite" else "Create a news draft"
    return (
        f"{prompt_goal} from the bounded fact pack below.\n"
        "Return JSON only. Do not wrap it in Markdown.\n"
        f"Required JSON object shape: {json.dumps(EXPECTED_JSON_SHAPE, ensure_ascii=True)}\n"
        "If facts are thin, set recommended_action to \"ignore\" or include uncertainties. "
        "Keep the body concise enough for a Discord embed.\n\n"
        f"Fact pack JSON:\n{json.dumps(fact_pack, ensure_ascii=True, indent=2)}"
    )


def _estimate_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return max(1, (len(system_prompt) + len(user_prompt) + 3) // 4)


def _openrouter_json_completion(
    settings: Settings,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.2,
) -> tuple[dict[str, Any], int]:
    if not settings.ai_drafts_enabled:
        raise DraftGenerationError("ai_drafts_disabled")
    if settings.llm_provider != "openrouter":
        raise DraftGenerationError(f"unsupported_llm_provider:{settings.llm_provider}")
    if not settings.openrouter_api_key.strip():
        raise DraftGenerationError("missing_openrouter_api_key")
    if not settings.openrouter_model.strip():
        raise DraftGenerationError("missing_openrouter_model")

    prompt_token_estimate = _estimate_prompt_tokens(system_prompt, user_prompt)
    request_payload = {
        "model": settings.openrouter_model.strip(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": OPENROUTER_REFERER,
            "X-Title": OPENROUTER_TITLE,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = response.read().decode("utf-8")
            status_code = getattr(response, "status", None) or getattr(response, "code", None)
        logger.info(
            "PNN AI request success: provider=%s model=%s status=%s",
            settings.llm_provider,
            settings.openrouter_model or None,
            status_code,
        )
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.warning(
            "PNN AI request failure: provider=%s model=%s status=%s fallback_reason=%s",
            settings.llm_provider,
            settings.openrouter_model or None,
            exc.code,
            f"openrouter_http_{exc.code}",
        )
        raise DraftGenerationError(
            f"openrouter_http_{exc.code}: {_clip_text(error_body, 400)}",
            prompt_token_estimate,
        ) from exc
    except (urllib.error.URLError, socket.timeout, TimeoutError) as exc:
        logger.warning(
            "PNN AI request failure: provider=%s model=%s fallback_reason=openrouter_connection_error",
            settings.llm_provider,
            settings.openrouter_model or None,
        )
        raise DraftGenerationError(
            f"openrouter_connection_error: {_clip_text(getattr(exc, 'reason', exc), 240)}",
            prompt_token_estimate,
        ) from exc

    try:
        data = json.loads(response_body)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        logger.warning(
            "PNN response parsing failed: provider=%s model=%s fallback_reason=openrouter_returned_unexpected_response",
            settings.llm_provider,
            settings.openrouter_model or None,
        )
        raise DraftGenerationError("openrouter_returned_unexpected_response", prompt_token_estimate) from exc

    try:
        parsed = _parse_json_object(content)
        logger.info(
            "PNN response parsing succeeded: provider=%s model=%s",
            settings.llm_provider,
            settings.openrouter_model or None,
        )
        return parsed, prompt_token_estimate
    except DraftGenerationError as exc:
        logger.warning(
            "PNN response parsing failed: provider=%s model=%s fallback_reason=%s",
            settings.llm_provider,
            settings.openrouter_model or None,
            exc.reason,
        )
        raise DraftGenerationError(exc.reason, prompt_token_estimate) from exc


def _parse_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise DraftGenerationError("model_returned_non_json")
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise DraftGenerationError("model_returned_invalid_json") from exc
    if not isinstance(parsed, dict):
        raise DraftGenerationError("model_returned_non_object_json")
    return parsed


def _normalize_confidence(value: Any) -> int:
    try:
        confidence = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise DraftGenerationError("model_returned_invalid_confidence") from exc
    if confidence < 0 or confidence > 100:
        raise DraftGenerationError("model_returned_confidence_out_of_range")
    return confidence


def _normalize_score(value: Any, field_name: str) -> int:
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise DraftGenerationError(f"model_returned_invalid_{field_name}") from exc
    if score < 0 or score > 100:
        raise DraftGenerationError(f"model_returned_{field_name}_out_of_range")
    return score


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise DraftGenerationError(f"model_returned_invalid_{field_name}")


def _normalize_choice(value: Any, allowed: set[str], field_name: str) -> str:
    normalized = _clean_text(value).lower()
    if normalized not in allowed:
        raise DraftGenerationError(f"model_returned_invalid_{field_name}")
    return normalized


def _normalize_uncertainties(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise DraftGenerationError("model_returned_invalid_uncertainties")
    return [_clip_text(item, 240) for item in value[:6] if _clean_text(item)]


def _contains_rule_breaking_claim(value: str) -> bool:
    return bool(RULE_BREAKING_PATTERN.search(value or ""))


def _similarity_ratio(left: str, right: str) -> float:
    def normalize(value: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", (value or "").lower()))

    normalized_left = normalize(left)
    normalized_right = normalize(right)
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right).ratio()


def _is_copy_like(generated: str, source: str) -> bool:
    return _similarity_ratio(generated, source) > 0.70


def _source_summary_from_message(message: dict[str, Any]) -> str:
    content = _clean_text(message.get("content"))
    if not content and message.get("attachments"):
        return "Attachment-only update"
    if not bool(message.get("staff_confirmed")) and _contains_rule_breaking_claim(content):
        return (
            "A player-chat message made an unconfirmed moderation allegation. "
            "Staff confirmation was not provided."
        )
    return shorten(content or "No text content provided", width=280, placeholder="...")


def _format_sources(source_links: list[dict[str, str]]) -> str:
    if not source_links:
        return "Source message link unavailable"
    lines = []
    for link in source_links[:5]:
        label = "Trigger" if link.get("role") == "triggering_message" else "Related"
        lines.append(f"- {label}: {link.get('url')}")
    return "\n".join(lines)


def _normalize_article_sections(value: Any) -> list[ArticleSection]:
    sections: list[ArticleSection] = []
    if isinstance(value, list):
        for item in value[:12]:
            if not isinstance(item, dict):
                continue
            title = _clip_text(item.get("title"), 120)
            content = _clip_multiline(item.get("content"), 1100)
            if title and content:
                sections.append(ArticleSection(title=title, content=content))
    return sections


def _article_body(article: ArticleContent, extra_blocks: Optional[list[tuple[str, str]]] = None) -> str:
    blocks = [article.summary]
    for section in article.sections:
        blocks.append(f"**{section.title}**\n{section.content}")
    if article.looking_ahead:
        blocks.append(f"**Looking Ahead**\n{article.looking_ahead}")
    for title, content in extra_blocks or []:
        if content:
            blocks.append(f"**{title}**\n{content}")
    return _clip_multiline("\n\n".join(blocks), MAX_DRAFT_BODY_CHARS)


def _sections_from_legacy_body(body: str) -> tuple[str, list[ArticleSection], str]:
    text = _clean_multiline(body)
    if not text:
        return "", [], ""

    heading_pattern = re.compile(r"^\*\*(?P<title>[^*\n]{1,120})\*\*\s*$")
    summary_lines: list[str] = []
    sections: list[ArticleSection] = []
    current_title: Optional[str] = None
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_title, current_lines
        if current_title and _clean_multiline("\n".join(current_lines)):
            sections.append(
                ArticleSection(
                    title=_clip_text(current_title, 120),
                    content=_clip_multiline("\n".join(current_lines), 1100),
                )
            )
        current_title = None
        current_lines = []

    for line in text.splitlines():
        match = heading_pattern.match(line.strip())
        if match:
            flush_section()
            current_title = match.group("title").strip()
            continue
        if current_title:
            current_lines.append(line)
        else:
            summary_lines.append(line)
    flush_section()

    skipped_titles = {"generation", "confidence", "risk level", "source note", "uncertainty notes", "uncertainties", "sources"}
    filtered_sections = [section for section in sections if section.title.strip().lower() not in skipped_titles]
    looking_ahead = ""
    kept_sections: list[ArticleSection] = []
    for section in filtered_sections:
        if section.title.strip().lower() in {"looking ahead", "what comes next"} and not looking_ahead:
            looking_ahead = section.content
        else:
            kept_sections.append(section)

    summary = _clip_multiline("\n".join(summary_lines), 700)
    if not summary and kept_sections:
        summary = _clip_multiline(kept_sections[0].content, 700)
    if not kept_sections and text:
        paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if item.strip()]
        if paragraphs:
            summary = summary or _clip_multiline(paragraphs[0], 700)
            for index, paragraph in enumerate(paragraphs[1:4], start=1):
                kept_sections.append(ArticleSection(title=f"Report {index}", content=_clip_multiline(paragraph, 1100)))
    return summary, kept_sections[:12], _clip_multiline(looking_ahead, 900)


def article_from_draft_content(draft: DraftContent) -> ArticleContent:
    if draft.article:
        return draft.article
    summary, sections, looking_ahead = _sections_from_legacy_body(draft.body)
    if not sections:
        sections = [ArticleSection(title="Report", content=_clip_multiline(draft.body, 1100))]
    return ArticleContent(
        headline=_clip_text(draft.title, 160) or "Prosperia News Update",
        summary=summary or _clip_multiline(draft.source_summary, 700) or "PNN has prepared this item for editor review.",
        sections=sections,
        looking_ahead=looking_ahead or "Editors should verify source details before publication.",
    )


def draft_content_from_article(article: dict[str, Any]) -> DraftContent:
    sections = _normalize_article_sections(article.get("sections"))
    content = ArticleContent(
        headline=_clip_text(article.get("headline"), 160) or "Prosperia News Update",
        summary=_clip_multiline(article.get("summary"), 900) or "PNN has prepared this item for editor review.",
        sections=sections or [ArticleSection(title="Report", content="No section content was provided.")],
        looking_ahead=_clip_multiline(article.get("looking_ahead"), 900),
    )
    return DraftContent(
        title=content.headline,
        body=_article_body(content),
        source_summary=_clip_text(content.summary, 280),
        article=content,
    )


def _article_from_journalist_payload(payload: dict[str, Any], verified_fact_pack: dict[str, Any]) -> ArticleContent:
    headline = _clip_text(payload.get("headline"), 160)
    summary = _clip_multiline(payload.get("summary"), 900)
    sections = _normalize_article_sections(payload.get("sections"))
    looking_ahead = _clip_multiline(payload.get("looking_ahead"), 900)

    legacy_what_happened = _clip_multiline(payload.get("what_happened"), 1200)
    legacy_why_it_matters = _clip_multiline(payload.get("why_it_matters"), 900)
    if not summary and legacy_what_happened:
        summary = legacy_what_happened
    if not sections:
        if legacy_what_happened:
            sections.append(ArticleSection(title="What Happened", content=legacy_what_happened))
        if legacy_why_it_matters:
            sections.append(ArticleSection(title="Why It Matters", content=legacy_why_it_matters))

    if not looking_ahead:
        uncertainties = _normalize_uncertainties(payload.get("uncertainties"))
        looking_ahead = (
            "Editors should watch for staff confirmation and additional source messages before publication."
            if uncertainties
            else "PNN will continue monitoring for further diplomatic, military, or economic developments."
        )

    if not headline or not summary or not sections:
        raise DraftGenerationError("journalist_returned_empty_draft")
    joined = "\n".join([headline, summary, *[section.content for section in sections], looking_ahead])
    if not verified_fact_pack.get("staff_confirmed") and _contains_rule_breaking_claim(joined):
        raise DraftGenerationError("journalist_returned_unconfirmed_rule_breaking_claim")

    return ArticleContent(
        headline=headline,
        summary=summary,
        sections=sections[:12],
        looking_ahead=looking_ahead,
    )


def _draft_from_ai_payload(payload: dict[str, Any], fact_pack: dict[str, Any]) -> DraftContent:
    headline = _clip_text(payload.get("headline"), 120)
    body = _clip_multiline(payload.get("body"), 2800)
    confidence = _normalize_confidence(payload.get("confidence"))
    risk_level = _normalize_choice(payload.get("risk_level"), {"low", "medium", "high"}, "risk_level")
    recommended_action = _normalize_choice(
        payload.get("recommended_action"),
        {"review", "ignore"},
        "recommended_action",
    )
    uncertainties = _normalize_uncertainties(payload.get("uncertainties"))

    if not headline or not body:
        raise DraftGenerationError("model_returned_empty_draft")
    if not fact_pack.get("staff_confirmed") and _contains_rule_breaking_claim(f"{headline}\n{body}"):
        raise DraftGenerationError("model_returned_unconfirmed_rule_breaking_claim")

    assessment = (
        f"**Assessment**\n"
        f"Confidence: {confidence}/100\n"
        f"Risk level: {risk_level}\n"
        f"Recommended action: {recommended_action}"
    )
    uncertainty_block = ""
    if uncertainties:
        uncertainty_lines = "\n".join(f"- {item}" for item in uncertainties)
        uncertainty_block = f"\n\n**Uncertainties**\n{uncertainty_lines}"

    source_note = ""
    if not fact_pack.get("staff_confirmed"):
        source_note = (
            "\n\n**Source note**\n"
            "Claims from Discord/player chat are unconfirmed unless staff confirmation is explicitly noted. "
            "This fact pack did not include staff confirmation."
        )

    sources = f"\n\n**Sources**\n{_format_sources(fact_pack.get('source_message_links') or [])}"
    formatted_body = _clip_multiline(
        f"{body}\n\n{assessment}{uncertainty_block}{source_note}{sources}",
        MAX_DRAFT_BODY_CHARS,
    )

    trigger = fact_pack.get("triggering_message") or {}
    source_summary = _source_summary_from_message(trigger)
    return DraftContent(title=headline, body=formatted_body, source_summary=source_summary)


def _group_messages_for_analysis(
    messages: list[dict[str, Any]],
    focus_message: Optional[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not messages:
        return [], []

    if focus_message:
        focus_keywords = _message_topic_keywords(focus_message)
        if not focus_keywords:
            focus_keywords = _extract_keywords(focus_message.get("content"))
        related: list[dict[str, Any]] = []
        for message in messages:
            candidate_keywords = _message_topic_keywords(message) or _extract_keywords(message.get("content"))
            if message.get("id") == focus_message.get("id") or focus_keywords.intersection(candidate_keywords):
                related.append(message)
        if focus_message not in related:
            related.insert(0, focus_message)
        topic_keywords = sorted(focus_keywords)[:8]
        return related[:25], topic_keywords

    groups: dict[str, list[dict[str, Any]]] = {}
    for message in messages:
        keywords = _message_topic_keywords(message)
        if not keywords:
            keywords = _extract_keywords(message.get("content"))
        for keyword in sorted(keywords)[:8]:
            groups.setdefault(keyword, []).append(message)

    if not groups:
        return messages[:25], []

    def group_rank(item: tuple[str, list[dict[str, Any]]]) -> tuple[int, int, int]:
        keyword, grouped_messages = item
        breaking_count = sum(1 for message in grouped_messages if _has_breaking_signal(message))
        event_count = sum(len(message.get("minecraft_events") or []) for message in grouped_messages)
        return (len(grouped_messages), breaking_count, event_count)

    top_keyword, top_messages = max(groups.items(), key=group_rank)
    topic_keywords = [top_keyword]
    shared_counts = {
        keyword: len(grouped)
        for keyword, grouped in groups.items()
        if keyword != top_keyword and any(message in top_messages for message in grouped)
    }
    topic_keywords.extend(
        keyword for keyword, _ in sorted(shared_counts.items(), key=lambda item: item[1], reverse=True)[:7]
    )
    return top_messages[:25], topic_keywords


def build_newsroom_fact_pack(
    messages: list[dict[str, Any]],
    settings: Settings,
    focus_message: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    related_messages, topic_keywords = _group_messages_for_analysis(messages, focus_message)
    source_links: list[dict[str, str]] = []
    minecraft_events: list[dict[str, str]] = []
    message_facts: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for message in related_messages[:25]:
        message_facts.append(_message_fact(message, MAX_RELATED_CONTENT_CHARS))
        link = _source_link("related_message", message)
        if link and link["url"] not in seen_urls:
            source_links.append(link)
            seen_urls.add(link["url"])
        minecraft_events.extend(_minecraft_events_for(message, "related_message"))

    staff_confirmed = any(bool(message.get("staff_confirmed")) for message in related_messages)
    breaking_signal = any(_has_breaking_signal(message) for message in related_messages)
    candidate_message_ids = [message["id"] for message in related_messages if message.get("id") is not None]
    highest_local_score = max([int(message.get("score") or 0) for message in related_messages] or [0])

    fact_pack = {
        "schema": "pnn_newsroom_fact_pack_v1",
        "analysis_window_minutes": settings.analysis_window_minutes,
        "minimum_related_messages": settings.min_related_messages,
        "news_score_threshold": settings.news_score_threshold,
        "ai_confidence_threshold": settings.ai_confidence_threshold,
        "staff_confirmed": staff_confirmed,
        "clear_breaking_signal": breaking_signal,
        "topic_keywords": topic_keywords,
        "candidate_message_ids": candidate_message_ids,
        "message_count": len(message_facts),
        "highest_local_score": highest_local_score,
        "rules": {
            "use_only_provided_facts": True,
            "player_chat_is_unconfirmed": True,
            "no_rule_breaking_accusations_without_staff_confirmed": True,
            "single_random_message_should_not_be_drafted_unless_clearly_breaking": True,
        },
        "messages": message_facts,
        "minecraft_events": minecraft_events[:MAX_MINECRAFT_EVENTS],
        "source_message_links": source_links[:12],
    }
    return _fit_newsroom_fact_pack(fact_pack)


def _fit_newsroom_fact_pack(fact_pack: dict[str, Any]) -> dict[str, Any]:
    while _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS and len(fact_pack["messages"]) > 1:
        fact_pack["messages"].pop()
    while _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS and fact_pack["minecraft_events"]:
        fact_pack["minecraft_events"].pop()
    while _fact_pack_size(fact_pack) > MAX_FACT_PACK_CHARS and fact_pack["source_message_links"]:
        fact_pack["source_message_links"].pop()
    return fact_pack


def _analyst_user_prompt(fact_pack: dict[str, Any]) -> str:
    return (
        "Analyze whether this rolling Discord/Minecraft window deserves a PNN draft.\n"
        "Return JSON only. Do not wrap it in Markdown.\n"
        f"Required JSON object shape: {json.dumps(ANALYST_JSON_SHAPE, ensure_ascii=True)}\n"
        "Use recommended_action=\"draft\" only when the facts support a real news item. "
        "Use \"monitor\" for developing chatter and \"ignore\" for isolated low-signal messages.\n\n"
        f"Fact pack JSON:\n{json.dumps(fact_pack, ensure_ascii=True, indent=2)}"
    )


def _analyst_decision_from_payload(payload: dict[str, Any], fact_pack: dict[str, Any]) -> AnalystDecision:
    decision = AnalystDecision(
        newsworthy=_normalize_bool(payload.get("newsworthy"), "newsworthy"),
        newsworthiness_score=_normalize_score(payload.get("newsworthiness_score"), "newsworthiness_score"),
        topic=_clip_text(payload.get("topic"), 120) or "Unclear topic",
        reason=_clip_text(payload.get("reason"), 500) or "No reason provided",
        recommended_action=_normalize_choice(
            payload.get("recommended_action"),
            {"ignore", "monitor", "draft"},
            "recommended_action",
        ),
        confidence=_normalize_confidence(payload.get("confidence")),
        facts=_normalize_uncertainties(payload.get("facts"))[:8],
        uncertainties=_normalize_uncertainties(payload.get("uncertainties")),
    )
    if not fact_pack.get("staff_confirmed"):
        joined = "\n".join([decision.topic, decision.reason, *decision.facts, *decision.uncertainties])
        if _contains_rule_breaking_claim(joined):
            raise DraftGenerationError("analyst_returned_unconfirmed_rule_breaking_claim")
    return decision


def _fallback_analyst_decision(fact_pack: dict[str, Any], fallback_reason: str) -> AnalystDecision:
    message_count = int(fact_pack.get("message_count") or 0)
    event_count = len(fact_pack.get("minecraft_events") or [])
    local_score = int(fact_pack.get("highest_local_score") or 0)
    breaking_signal = bool(fact_pack.get("clear_breaking_signal"))
    staff_confirmed = bool(fact_pack.get("staff_confirmed"))
    topic_keywords = fact_pack.get("topic_keywords") or []

    score = min(
        100,
        int(local_score * 0.45)
        + min(message_count, 6) * 9
        + min(event_count, 4) * 10
        + (18 if breaking_signal else 0)
        + (12 if staff_confirmed else 0),
    )
    confidence = min(
        100,
        38
        + min(message_count, 6) * 7
        + min(event_count, 4) * 6
        + (12 if staff_confirmed else 0)
        + (8 if breaking_signal else 0),
    )

    news_threshold = int(fact_pack.get("news_score_threshold") or 65)
    confidence_threshold = int(fact_pack.get("ai_confidence_threshold") or 60)
    if score >= news_threshold and confidence >= confidence_threshold and (message_count >= 2 or breaking_signal):
        action = "draft"
        newsworthy = True
    elif score >= 40:
        action = "monitor"
        newsworthy = False
    else:
        action = "ignore"
        newsworthy = False

    facts = []
    for message in fact_pack.get("messages") or []:
        channel = message.get("channel_name") or message.get("channel_id") or "unknown channel"
        content = _clip_text(message.get("content"), 180)
        if content:
            if not staff_confirmed and _contains_rule_breaking_claim(content):
                content = "A player-chat moderation allegation was present but was not staff-confirmed."
            facts.append(f"Unconfirmed player chat in #{channel}: {content}")
        if len(facts) >= 5:
            break
    for event in fact_pack.get("minecraft_events") or []:
        facts.append(f"Minecraft event supplied: {_jsonish(event, 220)}")
        if len(facts) >= 8:
            break

    if not facts:
        facts = ["No substantive facts were available in the rolling analysis window."]

    uncertainties = [f"AI analyst fallback used because {fallback_reason}."]
    if not staff_confirmed:
        uncertainties.append("No staff confirmation was included; player chat remains unconfirmed.")
    if message_count < 2 and not breaking_signal:
        uncertainties.append("Only one related message was found and no clear breaking signal was present.")

    return AnalystDecision(
        newsworthy=newsworthy,
        newsworthiness_score=score,
        topic=", ".join(topic_keywords[:3]) if topic_keywords else "Unclear topic",
        reason="Deterministic fallback analysis of the rolling Discord window.",
        recommended_action=action,
        confidence=confidence,
        facts=facts,
        uncertainties=uncertainties[:6],
    )


def _verified_fact_pack_from_analysis(fact_pack: dict[str, Any], decision: AnalystDecision) -> dict[str, Any]:
    trigger_message = (fact_pack.get("messages") or [{}])[0]
    return {
        "schema": "pnn_verified_fact_pack_v1",
        "topic": decision.topic,
        "newsworthiness_score": decision.newsworthiness_score,
        "confidence": decision.confidence,
        "staff_confirmed": bool(fact_pack.get("staff_confirmed")),
        "trigger_message": trigger_message,
        "recent_related_messages": fact_pack.get("messages") or [],
        "channel": trigger_message.get("channel_name") or trigger_message.get("channel_id"),
        "author": trigger_message.get("author_display") or trigger_message.get("author_id"),
        "source_links": fact_pack.get("source_message_links") or [],
        "server_events": fact_pack.get("minecraft_events") or [],
        "confidence_context": "player-chat only unless server events confirm it",
        "facts": decision.facts,
        "uncertainties": decision.uncertainties,
        "source_message_links": fact_pack.get("source_message_links") or [],
        "minecraft_events_available": bool(fact_pack.get("minecraft_events")),
        "anti_hallucination_rules": fact_pack.get("rules") or {},
    }


def analyze_news_window(
    messages: list[dict[str, Any]],
    settings: Settings,
    focus_message: Optional[dict[str, Any]] = None,
) -> NewsroomAnalysis:
    fact_pack = build_newsroom_fact_pack(messages, settings, focus_message=focus_message)
    related_count = int(fact_pack.get("message_count") or 0)
    logger.info(
        "PNN newsroom: related messages found=%s topic_keywords=%s window_minutes=%s",
        related_count,
        fact_pack.get("topic_keywords") or [],
        settings.analysis_window_minutes,
    )

    user_prompt = _analyst_user_prompt(fact_pack)
    prompt_token_estimate: Optional[int] = None
    analyst_used_ai = False
    fallback_reason: Optional[str] = None

    try:
        payload, prompt_token_estimate = _openrouter_json_completion(
            settings,
            ANALYST_SYSTEM_PROMPT,
            user_prompt,
            max_tokens=700,
            temperature=0.1,
        )
        decision = _analyst_decision_from_payload(payload, fact_pack)
        analyst_used_ai = True
    except DraftGenerationError as exc:
        prompt_token_estimate = exc.prompt_token_estimate
        fallback_reason = exc.reason
        decision = _fallback_analyst_decision(fact_pack, exc.reason)

    logger.info(
        "PNN newsroom: analyst decision=%s newsworthy=%s topic=%r confidence=%s ai_used=%s fallback_reason=%s",
        decision.recommended_action,
        decision.newsworthy,
        decision.topic,
        decision.confidence,
        analyst_used_ai,
        fallback_reason,
    )
    logger.info(
        "PNN newsroom: analyst score=%s threshold=%s confidence_threshold=%s",
        decision.newsworthiness_score,
        settings.news_score_threshold,
        settings.ai_confidence_threshold,
    )

    verified_fact_pack = _verified_fact_pack_from_analysis(fact_pack, decision)
    return NewsroomAnalysis(
        decision=decision,
        fact_pack=fact_pack,
        verified_fact_pack=verified_fact_pack,
        related_messages=fact_pack.get("messages") or [],
        topic_keywords=fact_pack.get("topic_keywords") or [],
        prompt_token_estimate=prompt_token_estimate,
        analyst_used_ai=analyst_used_ai,
        fallback_reason=fallback_reason,
    )


def should_create_news_draft(analysis: NewsroomAnalysis, settings: Settings) -> tuple[bool, str]:
    decision = analysis.decision
    if decision.recommended_action != "draft":
        return False, f"analyst_recommended_{decision.recommended_action}"
    if decision.newsworthiness_score < settings.news_score_threshold:
        return False, "newsworthiness_below_threshold"
    if decision.confidence < settings.ai_confidence_threshold:
        return False, "confidence_below_threshold"
    if (
        not settings.force_drafts
        and int(analysis.fact_pack.get("message_count") or 0) < settings.min_related_messages
        and not bool(analysis.fact_pack.get("clear_breaking_signal"))
    ):
        return False, "insufficient_related_messages"
    return True, "draft_gate_passed"


def _journalist_fact_pack(verified_fact_pack: dict[str, Any]) -> dict[str, Any]:
    trigger = verified_fact_pack.get("trigger_message") or {}
    return {
        "trigger_message": trigger.get("content") or "",
        "recent_related_messages": [
            {
                "content": message.get("content"),
                "channel": message.get("channel_name") or message.get("channel_id"),
                "author": message.get("author_display") or message.get("author_id"),
                "created_at": message.get("created_at"),
                "source_url": message.get("source_url"),
                "staff_confirmed": bool(message.get("staff_confirmed")),
            }
            for message in (verified_fact_pack.get("recent_related_messages") or [])[:8]
        ],
        "channel": verified_fact_pack.get("channel"),
        "author": verified_fact_pack.get("author"),
        "source_links": verified_fact_pack.get("source_links") or verified_fact_pack.get("source_message_links") or [],
        "server_events": verified_fact_pack.get("server_events") or [],
        "confidence_context": verified_fact_pack.get("confidence_context")
        or "player-chat only unless server events confirm it",
        "analyst": {
            "topic": verified_fact_pack.get("topic"),
            "newsworthiness_score": verified_fact_pack.get("newsworthiness_score"),
            "confidence": verified_fact_pack.get("confidence"),
            "facts": verified_fact_pack.get("facts") or [],
            "uncertainties": verified_fact_pack.get("uncertainties") or [],
            "staff_confirmed": bool(verified_fact_pack.get("staff_confirmed")),
        },
    }


def _trigger_message_text(verified_fact_pack: dict[str, Any]) -> str:
    trigger = verified_fact_pack.get("trigger_message") or {}
    return _clean_text(trigger.get("content") or "")


def _journalist_user_prompt(verified_fact_pack: dict[str, Any], extra_instruction: Optional[str] = None) -> str:
    instruction = f"\nAdditional instruction: {extra_instruction}\n" if extra_instruction else "\n"
    return (
        "Write a PNN draft from this structured fact pack only.\n"
        "Return JSON only. Do not wrap it in Markdown.\n"
        f"Required JSON object shape: {json.dumps(JOURNALIST_JSON_SHAPE, ensure_ascii=True)}\n"
        "The output must synthesize the facts and must not simply copy Discord message text. "
        "The headline, summary, sections, and looking_ahead must be original news-writing based on the facts. "
        "Sound like a serious geopolitical newsroom, not generic AI filler.\n"
        f"{instruction}\n"
        f"Fact pack JSON:\n{json.dumps(_journalist_fact_pack(verified_fact_pack), ensure_ascii=True, indent=2)}"
    )


def _fallback_journalist_payload(verified_fact_pack: dict[str, Any], fallback_reason: str) -> dict[str, Any]:
    topic = _clip_text(verified_fact_pack.get("topic"), 80) or "Developing Prosperia Story"
    facts = verified_fact_pack.get("facts") or []
    uncertainties = verified_fact_pack.get("uncertainties") or []
    fact_count = len(facts)
    return {
        "headline": f"PNN Watches {topic.title()}",
        "summary": (
            f"The PNN newsroom has identified a developing story around {topic}. "
            f"The analyst approved {fact_count} bounded fact(s) from the monitoring window for human review."
        ),
        "sections": [
            {
                "title": "What Happened",
                "content": (
                    f"The PNN newsroom has identified a developing story around {topic}. "
                    f"The analyst approved {fact_count} bounded fact(s) from the monitoring window for human review."
                ),
            },
            {
                "title": "Why It Matters",
                "content": "The item may affect diplomatic, military, or economic decisions if the reported facts are confirmed.",
            },
        ],
        "looking_ahead": "Editors should verify the source window and watch for corroborating messages before publication.",
        "confidence": verified_fact_pack.get("confidence", 0),
        "risk_level": "medium",
        "source_note": "fallback_used=true. Generated by fallback template, not AI.",
        "uncertainties": [*uncertainties[:4], f"Fallback reason: {fallback_reason}."],
    }


def _draft_from_journalist_payload(
    payload: dict[str, Any],
    verified_fact_pack: dict[str, Any],
    generated_by_ai: bool,
) -> DraftContent:
    article = _article_from_journalist_payload(payload, verified_fact_pack)
    confidence = _normalize_confidence(payload.get("confidence"))
    risk_level = _normalize_choice(payload.get("risk_level"), {"low", "medium", "high"}, "risk_level")
    source_note = _clip_multiline(payload.get("source_note"), 700)
    uncertainties = _normalize_uncertainties(payload.get("uncertainties"))

    if not source_note:
        raise DraftGenerationError("journalist_returned_empty_draft")

    uncertainty_block = "None listed."
    if uncertainties:
        uncertainty_block = "\n".join(f"- {note}" for note in uncertainties)
    sources = _format_sources(verified_fact_pack.get("source_message_links") or [])
    fallback_note = ""
    if not generated_by_ai:
        fallback_note = "**Generation**\nfallback_used=true\nGenerated by fallback template, not AI.\n\n"
    if not verified_fact_pack.get("staff_confirmed"):
        source_note = (
            f"{source_note}\n"
            "Player-chat claims are unconfirmed unless staff confirmation is explicitly noted."
        )

    body = _article_body(
        article,
        [
            ("Generation", fallback_note.replace("**Generation**\n", "").strip()),
            ("Confidence", f"{confidence}/100"),
            ("Risk level", risk_level),
            ("Source note", source_note),
            ("Uncertainty notes", uncertainty_block),
            ("Sources", sources),
        ],
    )
    source_summary = _clip_text("; ".join(verified_fact_pack.get("facts") or []), 280) or "Analyst-approved fact pack"
    return DraftContent(title=article.headline, body=body, source_summary=source_summary, article=article)


def _journalist_payload_copies_trigger(payload: dict[str, Any], trigger_message: str) -> bool:
    section_text = " ".join(
        _clean_text(section.get("content"))
        for section in (payload.get("sections") or [])
        if isinstance(section, dict)
    )
    return (
        _is_copy_like(_clean_text(payload.get("headline")), trigger_message)
        or _is_copy_like(_clean_text(payload.get("summary") or payload.get("what_happened")), trigger_message)
        or _is_copy_like(section_text, trigger_message)
    )


def _fallback_allowed_reason(reason: str) -> bool:
    return (
        reason == "ai_drafts_disabled"
        or reason == "missing_openrouter_api_key"
        or reason.startswith("openrouter_http_")
        or reason.startswith("openrouter_connection_error")
        or reason in {"openrouter_returned_unexpected_response", "json_parsing_failed_twice"}
    )


def generate_journalist_draft(analysis: NewsroomAnalysis, settings: Settings) -> DraftContent:
    verified_fact_pack = analysis.verified_fact_pack
    trigger_message = _trigger_message_text(verified_fact_pack)
    logger.info(
        "PNN newsroom: journalist config AI_DRAFTS_ENABLED=%s LLM_PROVIDER=%s model=%s",
        settings.ai_drafts_enabled,
        settings.llm_provider,
        settings.openrouter_model or None,
    )

    if not settings.ai_drafts_enabled:
        fallback_reason = "ai_drafts_disabled"
        logger.warning(
            "PNN newsroom: journalist LLM attempted=false succeeded=false fallback_used=true fallback_template_used=true fallback_reason=%s",
            fallback_reason,
        )
        return _draft_from_journalist_payload(
            _fallback_journalist_payload(verified_fact_pack, fallback_reason),
            verified_fact_pack,
            generated_by_ai=False,
        )

    if not settings.openrouter_api_key.strip():
        fallback_reason = "missing_openrouter_api_key"
        logger.warning(
            "PNN newsroom: journalist LLM attempted=false succeeded=false fallback_used=true fallback_template_used=true fallback_reason=%s",
            fallback_reason,
        )
        return _draft_from_journalist_payload(
            _fallback_journalist_payload(verified_fact_pack, fallback_reason),
            verified_fact_pack,
            generated_by_ai=False,
        )

    parse_failure_count = 0
    extra_instruction: Optional[str] = None
    last_error: Optional[DraftGenerationError] = None

    for attempt in range(1, 3):
        user_prompt = _journalist_user_prompt(verified_fact_pack, extra_instruction=extra_instruction)
        try:
            logger.info(
                "PNN newsroom: journalist LLM attempted=true attempt=%s provider=%s model=%s",
                attempt,
                settings.llm_provider,
                settings.openrouter_model or None,
            )
            payload, prompt_token_estimate = _openrouter_json_completion(
                settings,
                JOURNALIST_SYSTEM_PROMPT,
                user_prompt,
                max_tokens=900,
                temperature=0.2,
            )
            if _journalist_payload_copies_trigger(payload, trigger_message):
                logger.warning(
                    "PNN newsroom: journalist anti-copy retry triggered attempt=%s headline_similarity=%.3f what_happened_similarity=%.3f",
                    attempt,
                    _similarity_ratio(_clean_text(payload.get("headline")), trigger_message),
                    _similarity_ratio(_clean_text(payload.get("what_happened")), trigger_message),
                )
                if attempt == 1:
                    extra_instruction = "Do not copy the original message. Synthesize it into a neutral news report."
                    continue
                raise DraftGenerationError("journalist_copied_trigger_message", prompt_token_estimate)

            draft = _draft_from_journalist_payload(payload, verified_fact_pack, generated_by_ai=True)
            logger.info(
                "PNN newsroom: journalist LLM attempted=true succeeded=true fallback_used=false fallback_template_used=false model=%s prompt_token_estimate=%s",
                settings.openrouter_model or None,
                prompt_token_estimate,
            )
            return draft
        except DraftGenerationError as exc:
            last_error = exc
            if exc.reason.startswith("model_returned_") or exc.reason == "journalist_returned_empty_draft":
                parse_failure_count += 1
                logger.warning(
                    "PNN newsroom: journalist JSON/schema failure attempt=%s reason=%s",
                    attempt,
                    exc.reason,
                )
                if attempt == 1:
                    extra_instruction = (
                        "Return only valid JSON matching the required schema. "
                        "Do not copy the original message. Synthesize it into a neutral news report."
                    )
                    continue
                break
            if _fallback_allowed_reason(exc.reason):
                logger.warning(
                    "PNN newsroom: journalist LLM attempted=true succeeded=false fallback_used=true fallback_template_used=true fallback_reason=%s",
                    exc.reason,
                )
                return _draft_from_journalist_payload(
                    _fallback_journalist_payload(verified_fact_pack, exc.reason),
                    verified_fact_pack,
                    generated_by_ai=False,
                )
            logger.error(
                "PNN newsroom: journalist LLM attempted=true succeeded=false fallback_template_used=false reason=%s",
                exc.reason,
            )
            raise

    fallback_reason = "json_parsing_failed_twice" if parse_failure_count >= 2 else (last_error.reason if last_error else "unknown")
    if _fallback_allowed_reason(fallback_reason):
        logger.warning(
            "PNN newsroom: journalist LLM attempted=true succeeded=false fallback_used=true fallback_template_used=true fallback_reason=%s",
            fallback_reason,
        )
        return _draft_from_journalist_payload(
            _fallback_journalist_payload(verified_fact_pack, fallback_reason),
            verified_fact_pack,
            generated_by_ai=False,
        )

    reason = fallback_reason or "journalist_generation_failed"
    logger.error(
        "PNN newsroom: journalist LLM attempted=true succeeded=false fallback_template_used=false reason=%s",
        reason,
    )
    raise DraftGenerationError(reason)


def _title_from_message(message: dict[str, Any]) -> str:
    content = _clean_text(message.get("content"))
    channel = message.get("channel_name") or "Discord"
    if content:
        first_sentence = content.split(". ")[0]
        title = shorten(first_sentence, width=88, placeholder="...")
        return title[:1].upper() + title[1:]
    return f"Update from #{channel}"


def generate_template_draft(
    message: dict[str, Any],
    score_result: ScoreResult,
    variant: str = "standard",
) -> DraftContent:
    content = _clean_text(message.get("content"))
    channel = message.get("channel_name") or "unknown-channel"
    author = message.get("author_display") or message.get("author_id") or "unknown author"
    source_url = message.get("jump_url") or "Source message link unavailable"
    source_summary = _source_summary_from_message(message)
    reasons = "; ".join(score_result.reasons)
    confirmation_note = (
        "Staff confirmation was included with the source."
        if message.get("staff_confirmed")
        else "This remains unconfirmed player-chat reporting until a reviewer verifies it."
    )

    title = _title_from_message(message)
    if not message.get("staff_confirmed") and _contains_rule_breaking_claim(content):
        title = "Unconfirmed moderation allegation requires review"

    if variant == "rewrite":
        body = (
            f"**Generation**\nfallback_used=true\nGenerated by fallback template, not AI.\n\n"
            f"**Update**\n{source_summary}\n\n"
            f"**Context**\nThis was flagged from #{channel} with a newsworthiness score of "
            f"{score_result.score}/100. Signals: {reasons}.\n\n"
            f"**Source note**\n{confirmation_note}\n\n"
            f"**Source**\n{source_url}"
        )
    else:
        body = (
            f"**Generation**\nfallback_used=true\nGenerated by fallback template, not AI.\n\n"
            f"**What happened**\n{source_summary}\n\n"
            f"**Why it matters**\nThis Discord message was flagged as potentially newsworthy "
            f"because: {reasons}.\n\n"
            f"**Attribution**\nOriginally posted by {author} in #{channel}.\n\n"
            f"**Source note**\n{confirmation_note}\n\n"
            f"**Source**\n{source_url}"
        )

    return DraftContent(title=title, body=body, source_summary=source_summary)
