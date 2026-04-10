"""
retriever.py – Tìm kiếm chunk liên quan bằng vector search trong Qdrant.

Hỗ trợ lọc theo metadata:
  - source        : tên file hoặc domain URL
  - content_type  : "web", "pdf", "text", ...
  - language      : "vi", "en", ...
  - tags          : danh sách nhãn
  - doc_id        : UUID tài liệu cụ thể
  - author        : tên tác giả

Trả về danh sách RetrievedChunk, có thể dùng trực tiếp trong generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from rag.config import settings
from rag.schema.metadata import DocumentMetadata, RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# MetadataFilter – helper để xây dựng Qdrant Filter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MetadataFilter:
    """
    Bộ lọc metadata linh hoạt.

    Ví dụ:
        f = MetadataFilter(source="https://example.com", language="vi")
        chunks = retriever.search("câu hỏi", filters=f)
    """
    source: str | None = None
    content_type: str | None = None
    language: str | None = None
    doc_id: str | None = None
    author: str | None = None
    tags: list[str] = field(default_factory=list)

    def to_qdrant_filter(self):
        """Chuyển sang qdrant_client.models.Filter."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

        conditions = []

        if self.source:
            conditions.append(
                FieldCondition(key="source", match=MatchValue(value=self.source))
            )
        if self.content_type:
            conditions.append(
                FieldCondition(key="content_type", match=MatchValue(value=self.content_type))
            )
        if self.language:
            conditions.append(
                FieldCondition(key="language", match=MatchValue(value=self.language))
            )
        if self.doc_id:
            conditions.append(
                FieldCondition(key="doc_id", match=MatchValue(value=self.doc_id))
            )
        if self.author:
            conditions.append(
                FieldCondition(key="author", match=MatchValue(value=self.author))
            )
        if self.tags:
            for tag in self.tags:
                conditions.append(
                    FieldCondition(key="tags", match=MatchValue(value=tag))
                )

        if not conditions:
            return None
        return Filter(must=conditions)


# ─────────────────────────────────────────────────────────────────────────────
# QdrantRetriever
# ─────────────────────────────────────────────────────────────────────────────

class QdrantRetriever:
    """
    Tìm kiếm chunk liên quan bằng vector similarity search.

    Ví dụ:
        retriever = QdrantRetriever()
        results = retriever.search(
            "VinFast VF8 có pin bao nhiêu?",
            top_k=5,
            filters=MetadataFilter(language="vi"),
        )
        for chunk in results:
            print(chunk.score, chunk.text[:100])
    """

    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise ImportError("Cài đặt `qdrant-client`") from e

        self.collection_name = collection_name or settings.qdrant_collection
        self._client = QdrantClient(
            url=qdrant_url or settings.qdrant_url,
            api_key=qdrant_api_key or settings.qdrant_api_key,
        )
        self._openai_client = None  # lazy load

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY chưa được cấu hình.")
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
            logger.info(f"OpenAI Embedding client sẵn sàng | model={settings.embedding_model}")
        return self._openai_client

    def _embed_query(self, query: str) -> list[float]:
        """Embed một câu query bằng OpenAI Embedding API."""
        client = self._get_openai_client()
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=query,
        )
        return response.data[0].embedding

    # ── Search ────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Tìm kiếm chunk liên quan đến câu hỏi.

        Args:
            query          : Câu hỏi hoặc cụm từ cần tìm kiếm
            top_k          : Số chunk trả về tối đa
            filters        : Bộ lọc metadata (tuỳ chọn)
            score_threshold: Loại bỏ chunk có score thấp hơn ngưỡng

        Returns:
            Danh sách RetrievedChunk sắp xếp theo score giảm dần
        """
        top_k = top_k or settings.retrieval_top_k
        threshold = score_threshold or settings.retrieval_score_threshold

        # Embed query bằng OpenAI
        query_vector = self._embed_query(query)

        qdrant_filter = filters.to_qdrant_filter() if filters else None

        logger.debug(
            f"Vector search | top_k={top_k} | "
            f"filter={'yes' if qdrant_filter else 'no'}"
        )

        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
            score_threshold=threshold,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results.points:
            payload = hit.payload
            text = payload.pop("text", "")

            try:
                meta = DocumentMetadata.from_qdrant_payload(payload)
            except Exception as e:
                logger.warning(f"Không thể parse metadata cho point {hit.id}: {e}")
                continue

            chunks.append(
                RetrievedChunk(
                    id=str(hit.id),
                    text=text,
                    score=hit.score,
                    metadata=meta,
                )
            )

        logger.debug(f"Tìm thấy {len(chunks)} chunk(s) có score >= {threshold}")
        return chunks

    # ── Scroll / Get all ──────────────────────────────────────

    def scroll_by_source(
        self, source: str, limit: int = 100
    ) -> list[RetrievedChunk]:
        """Lấy tất cả chunk thuộc một source cụ thể (không dùng vector)."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        points, _ = self._client.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        chunks = []
        for point in points:
            payload = dict(point.payload)
            text = payload.pop("text", "")
            try:
                meta = DocumentMetadata.from_qdrant_payload(payload)
                chunks.append(RetrievedChunk(id=str(point.id), text=text, score=1.0, metadata=meta))
            except Exception:
                pass
        return chunks
