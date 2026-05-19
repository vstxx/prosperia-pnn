from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"
if ROOT_ENV.exists():
    load_dotenv(ROOT_ENV)
load_dotenv()


def _bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


@dataclass(frozen=True)
class Settings:
    app_name: str = "Prosperia News Network"
    database_path: str = os.getenv("DATABASE_PATH", "data/pnn.sqlite3")
    pnn_shared_secret: str = os.getenv("PNN_SHARED_SECRET", "")
    internal_api_key: str = os.getenv("INTERNAL_API_KEY", os.getenv("PNN_SHARED_SECRET", ""))
    news_score_threshold: int = _int_env("NEWS_SCORE_THRESHOLD", 65)
    ai_confidence_threshold: int = _int_env("AI_CONFIDENCE_THRESHOLD", 60)
    analysis_window_minutes: int = _int_env("ANALYSIS_WINDOW_MINUTES", 10)
    min_related_messages: int = _int_env("MIN_RELATED_MESSAGES", 2)
    force_drafts: bool = _bool_env("FORCE_DRAFTS", _bool_env("PNN_FORCE_DRAFTS", False))
    pnn_bot_callback_url: str = os.getenv(
        "PNN_BOT_CALLBACK_URL",
        "https://prosperia-pnn.onrender.com/callbacks/drafts",
    )
    bot_review_endpoint: str = os.getenv(
        "BOT_REVIEW_ENDPOINT",
        os.getenv("PNN_BOT_CALLBACK_URL", "http://localhost:3000/pnn-review"),
    )
    ai_drafts_enabled: bool = _bool_env("AI_DRAFTS_ENABLED", False)
    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter").strip().lower()
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

    @property
    def pnn_force_drafts(self) -> bool:
        return self.force_drafts


def get_settings() -> Settings:
    return Settings()
