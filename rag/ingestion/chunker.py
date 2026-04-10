"""
chunker.py – Chia văn bản dài thành các chunk nhỏ hơn.

Sử dụng RecursiveCharacterTextSplitter từ langchain-text-splitters.
Mỗi chunk giữ nguyên text và có thể gắn metadata vị trí.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from rag.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Data class: một chunk văn bản
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """Một đoạn văn bản đã được chia nhỏ."""
    text: str
    chunk_index: int
    total_chunks: int
    char_start: int = 0    # Vị trí ký tự bắt đầu trong tài liệu gốc
    char_end: int = 0      # Vị trí ký tự kết thúc


# ─────────────────────────────────────────────────────────────────────────────
# TextChunker
# ─────────────────────────────────────────────────────────────────────────────

class TextChunker:
    """
    Chia văn bản thành các chunk với kích thước cấu hình được.

    Ví dụ:
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        chunks = chunker.split("Nội dung văn bản dài...")
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Phân tách theo thứ tự ưu tiên:
        # đoạn văn > dòng > câu > từ > ký tự
        self.separators = separators or ["\n\n", "\n", "。", ".", "!", "?", " ", ""]

        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as e:
            raise ImportError(
                "Cài đặt `langchain-text-splitters` để dùng TextChunker"
            ) from e

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            f"TextChunker init: chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def split(self, text: str) -> list[TextChunk]:
        """
        Chia văn bản thành danh sách TextChunk.

        Args:
            text: Văn bản gốc cần chia.

        Returns:
            Danh sách TextChunk, mỗi chunk có text + vị trí.
        """
        if not text.strip():
            logger.warning("Văn bản rỗng, không thể chia chunk.")
            return []

        raw_chunks = self._splitter.split_text(text)
        total = len(raw_chunks)

        chunks: list[TextChunk] = []
        cursor = 0
        for idx, chunk_text in enumerate(raw_chunks):
            start = text.find(chunk_text, max(0, cursor - self.chunk_overlap))
            end = start + len(chunk_text)
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    chunk_index=idx,
                    total_chunks=total,
                    char_start=start if start != -1 else cursor,
                    char_end=end if start != -1 else cursor + len(chunk_text),
                )
            )
            if start != -1:
                cursor = end

        logger.debug(f"Tài liệu được chia thành {total} chunks")
        return chunks

    def split_batch(self, texts: list[str]) -> list[list[TextChunk]]:
        """Chia nhiều tài liệu cùng lúc."""
        return [self.split(t) for t in texts]
