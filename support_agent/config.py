"""Configuration: every knob comes from the environment, documented in .env.example."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (stdlib only): KEY=VALUE lines, no interpolation."""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            os.environ.setdefault(key, value)


@dataclass
class Settings:
    # Storage backend: "sqlite" (offline default) or "postgres".
    backend: str = "sqlite"
    db_path: str = "data/support.db"
    database_url: str = ""  # postgres DSN for the live backend

    # Retrieval / answering.
    top_k: int = 3
    confidence_threshold: float = 0.12
    embedder: str = "hash"  # "hash" (offline) | "openai" (live, documented)
    embedding_dim: int = 512
    chunk_max_chars: int = 1200
    chunk_overlap_chars: int = 150

    # Optional live LLM rephrasing (offline composer is extractive; LLM only rewords).
    llm_provider: str = ""  # e.g. "openai"
    llm_model: str = ""
    llm_api_key: str = ""

    # Telegram transport.
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Ticket follow-ups.
    support_contact: str = ""

    extra: dict = field(default_factory=dict, repr=False)


def load_settings(env: dict | None = None, dotenv_path: str = ".env") -> Settings:
    if env is None:
        _load_dotenv(dotenv_path)
        env = os.environ

    def get(name: str, default: str = "") -> str:
        return str(env.get(name, default))

    return Settings(
        backend=get("SUPPORT_BACKEND", "sqlite").lower(),
        db_path=get("SUPPORT_DB_PATH", "data/support.db"),
        database_url=get("DATABASE_URL", ""),
        top_k=int(get("SUPPORT_TOP_K", "3")),
        confidence_threshold=float(get("SUPPORT_THRESHOLD", "0.12")),
        embedder=get("SUPPORT_EMBEDDER", "hash").lower(),
        embedding_dim=int(get("SUPPORT_EMBEDDING_DIM", "512")),
        chunk_max_chars=int(get("SUPPORT_CHUNK_MAX_CHARS", "1200")),
        chunk_overlap_chars=int(get("SUPPORT_CHUNK_OVERLAP", "150")),
        llm_provider=get("LLM_PROVIDER", "").lower(),
        llm_model=get("LLM_MODEL", ""),
        llm_api_key=get("LLM_API_KEY", ""),
        telegram_bot_token=get("TELEGRAM_BOT_TOKEN", ""),
        telegram_chat_id=get("TELEGRAM_CHAT_ID", ""),
        support_contact=get("SUPPORT_CONTACT", ""),
    )
