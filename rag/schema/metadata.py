"""
metadata.py – Định nghĩa schema metadata cho từng chunk tài liệu.

Metadata tối thiểu (required):
  source       : URL hoặc đường dẫn tệp gốc
  doc_id       : UUID định danh tài liệu
  chunk_index  : Vị trí chunk trong tài liệu (0-indexed)
  total_chunks : Tổng số chunk của tài liệu
  created_at   : Thời điểm ingest (UTC)

Metadata mở rộng (optional):
  title        : Tiêu đề tài liệu
  language     : Mã ngôn ngữ, vd: "vi", "en"
  content_type : "web" | "pdf" | "text" | "docx" | "markdown"
  author       : Tác giả
  tags         : Danh sách nhãn phân loại
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# ContentType enum
# ─────────────────────────────────────────────────────────────────────────────
VALID_CONTENT_TYPES = {"web", "pdf", "text", "docx", "markdown"}


# ─────────────────────────────────────────────────────────────────────────────
# Core metadata schema
# ─────────────────────────────────────────────────────────────────────────────
class DocumentMetadata(BaseModel):
    """
    Metadata gắn kèm với từng chunk khi lưu vào Qdrant.
    Được serialize thành payload dict.
    """

    # ── Required ──────────────────────────────────────────────
    source: str = Field(
        ...,
        description="URL hoặc đường dẫn tệp gốc của tài liệu",
        examples=["https://example.com/article", "/data/docs/report.pdf"],
    )
    doc_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="UUID định danh tài liệu (tất cả chunk cùng doc chia sẻ ID này)",
    )
    chunk_index: int = Field(
        default=0,
        ge=0,
        description="Vị trí chunk trong tài liệu (0-indexed)",
    )
    total_chunks: int = Field(
        default=1,
        ge=1,
        description="Tổng số chunk của tài liệu",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Thời điểm ingest (ISO-8601, UTC)",
    )

    # ── Optional ──────────────────────────────────────────────
    title: str | None = Field(
        default=None,
        description="Tiêu đề tài liệu",
    )
    language: str | None = Field(
        default=None,
        description='Mã ngôn ngữ ISO 639-1, vd: "vi", "en"',
        examples=["vi", "en", "ja"],
    )
    content_type: str = Field(
        default="text",
        description=f"Loại nội dung: {VALID_CONTENT_TYPES}",
    )
    author: str | None = Field(
        default=None,
        description="Tác giả / nguồn phát hành",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Danh sách nhãn phân loại tự do",
        examples=[["vinfast", "xe-dien", "review"]],
    )

    # ── Validator ─────────────────────────────────────────────
    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        if v not in VALID_CONTENT_TYPES:
            raise ValueError(
                f"content_type phải là một trong: {VALID_CONTENT_TYPES}"
            )
        return v

    # ── Helpers ───────────────────────────────────────────────
    def to_qdrant_payload(self) -> dict[str, Any]:
        """Serialize sang dict dùng làm Qdrant point payload."""
        return self.model_dump(exclude_none=False)

    @classmethod
    def from_qdrant_payload(cls, payload: dict[str, Any]) -> "DocumentMetadata":
        """Khôi phục metadata từ Qdrant point payload."""
        return cls(**payload)


# ─────────────────────────────────────────────────────────────────────────────
# Input model – dùng khi người dùng nạp tài liệu vào hệ thống
# ─────────────────────────────────────────────────────────────────────────────
class DocumentInput(BaseModel):
    """
    Đầu vào để ingest một tài liệu. Người dùng cung cấp nội dung
    và metadata tùy chọn; hệ thống sẽ tự điền các trường còn lại.
    """

    content: str = Field(..., description="Nội dung văn bản thô")
    source: str = Field(..., description="URL hoặc đường dẫn tệp gốc")
    title: str | None = None
    language: str | None = None
    content_type: str = "text"
    author: str | None = None
    tags: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Retrieved chunk – kết quả trả về từ retriever
# ─────────────────────────────────────────────────────────────────────────────
class RetrievedChunk(BaseModel):
    """Một chunk được tìm thấy bởi vector search, kèm score và metadata."""

    id: str = Field(..., description="UUID của point trong Qdrant")
    text: str = Field(..., description="Nội dung văn bản của chunk")
    score: float = Field(..., description="Cosine similarity score [0, 1]")
    metadata: DocumentMetadata
