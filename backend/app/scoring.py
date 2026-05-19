from __future__ import annotations

from dataclasses import dataclass
import re


URL_PATTERN = re.compile(r"https?://", re.IGNORECASE)

KEYWORD_WEIGHTS: dict[str, int] = {
    "breaking": 24,
    "urgent": 20,
    "outage": 18,
    "incident": 18,
    "launch": 16,
    "released": 14,
    "announced": 14,
    "approved": 14,
    "blocked": 13,
    "security": 16,
    "exploit": 18,
    "partnership": 14,
    "funding": 14,
    "deadline": 12,
    "migration": 12,
    "downtime": 16,
    "confirmed": 12,
    "policy": 12,
    "pricing": 12,
    "revenue": 12,
    "customers": 10,
    "users": 10,
}

LOW_SIGNAL_WORDS = {
    "lol",
    "lmao",
    "thanks",
    "thank you",
    "gm",
    "gn",
    "ok",
    "okay",
    "nice",
    "cool",
}


@dataclass(frozen=True)
class ScoreResult:
    score: int
    reasons: list[str]


def score_message(content: str, attachments: list[dict]) -> ScoreResult:
    text = (content or "").strip()
    lowered = text.lower()
    score = 0
    reasons: list[str] = []

    if len(text) >= 240:
        score += 16
        reasons.append("substantial message length")
    elif len(text) >= 100:
        score += 10
        reasons.append("meaningful message length")
    elif len(text) >= 40:
        score += 5
        reasons.append("some context provided")

    matched_keywords = []
    for keyword, weight in KEYWORD_WEIGHTS.items():
        if keyword in lowered:
            matched_keywords.append(keyword)
            score += weight

    if matched_keywords:
        reasons.append("matched news terms: " + ", ".join(sorted(set(matched_keywords))[:6]))

    if URL_PATTERN.search(text):
        score += 8
        reasons.append("contains a link")

    if attachments:
        score += 8
        reasons.append("includes attachment evidence")

    if "?" in text and len(text) < 120:
        score -= 8
        reasons.append("short question, likely not a draft on its own")

    if lowered in LOW_SIGNAL_WORDS or any(lowered.startswith(word + " ") for word in LOW_SIGNAL_WORDS):
        score -= 16
        reasons.append("low-signal conversational wording")

    if not text and attachments:
        score = max(score, 18)
        reasons.append("attachment-only message needs manual context")

    score = max(0, min(100, score))
    if not reasons:
        reasons.append("no strong news signals found")

    return ScoreResult(score=score, reasons=reasons)
