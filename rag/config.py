"""
config.py – Cấu hình toàn hệ thống, đọc từ biến môi trường / .env
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Qdrant ────────────────────────────────────────────────
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection: str = Field(default="rag_documents")

    # ── Firecrawl (Self-hosted) ──────────────────────────────
    firecrawl_api_url: str = Field(default="http://localhost:3002")
    firecrawl_api_key: str = Field(default="none")

    # ── LLM (Gemini & OpenAI) ─────────────────────────────────
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")
    openai_chat_model: str = Field(default="gpt-4o-mini")

    # ── Embedding (OpenAI) ────────────────────────────────────
    openai_api_key: str | None = Field(default=None)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Vector dimension – 1536 cho text-embedding-3-small, 3072 cho text-embedding-3-large",
    )

    # ── Chunking ──────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=64)

    # ── Retrieval ─────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5)
    retrieval_score_threshold: float = Field(default=0.3)

    # ── Logging ───────────────────────────────────────────────
    log_level: str = Field(default="INFO")


# Singleton instance
settings = Settings()
