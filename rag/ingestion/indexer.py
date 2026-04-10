"""
indexer.py – Embed văn bản và upsert vào Qdrant.

Workflow:
  1. Nhận DocumentInput + danh sách TextChunk
  2. Embed tất cả chunk bằng OpenAI Embedding API
  3. Tạo Qdrant PointStruct với vector + payload (metadata)
  4. Upsert vào collection

Collection được tạo tự động nếu chưa tồn tại.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from loguru import logger
from tqdm import tqdm

from rag.config import settings
from rag.schema.metadata import DocumentInput, DocumentMetadata
from rag.ingestion.chunker import TextChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Embedding helper (lazy-loaded client)
# ─────────────────────────────────────────────────────────────────────────────

_openai_client = None


def _get_openai_client():
    """Khởi tạo OpenAI client (singleton)."""
    global _openai_client
    if _openai_client is None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Cài đặt `openai` để dùng OpenAI Embedding"
            ) from e

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY chưa được cấu hình. "
                "Thêm vào file .env hoặc biến môi trường."
            )
        _openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.info(f"OpenAI client sẵn sàng | model={settings.embedding_model}")
    return _openai_client


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách text bằng OpenAI Embedding API."""
    client = _get_openai_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


# ─────────────────────────────────────────────────────────────────────────────
# QdrantIndexer
# ─────────────────────────────────────────────────────────────────────────────

class QdrantIndexer:
    """
    Nhúng văn bản và lưu vào Qdrant.

    Ví dụ:
        indexer = QdrantIndexer()
        indexer.index(document_input, chunks)
    """

    def __init__(
        self,
        collection_name: str | None = None,
        qdrant_url: str | None = None,
        qdrant_api_key: str | None = None,
    ) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as e:
            raise ImportError("Cài đặt `qdrant-client`") from e

        self.collection_name = collection_name or settings.qdrant_collection
        self._client = QdrantClient(
            url=qdrant_url or settings.qdrant_url,
            api_key=qdrant_api_key or settings.qdrant_api_key,
        )
        self._ensure_collection()

    # ── Collection management ──────────────────────────────────

    def _ensure_collection(self) -> None:
        """Tạo collection nếu chưa tồn tại."""
        from qdrant_client.models import Distance, VectorParams

        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Tạo collection: {self.collection_name}")
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.embedding_dimension,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.debug(f"Collection đã tồn tại: {self.collection_name}")

    # ── Indexing ───────────────────────────────────────────────

    def index(
        self,
        document: DocumentInput,
        chunks: list[TextChunk],
        doc_id: str | None = None,
        batch_size: int = 64,
    ) -> list[str]:
        """
        Embed và upsert toàn bộ chunk của một tài liệu.

        Args:
            document : Thông tin tài liệu gốc (source, title, …)
            chunks   : Danh sách TextChunk từ chunker
            doc_id   : Nếu None sẽ tạo UUID mới
            batch_size: Số chunk xử lý mỗi lần gọi model

        Returns:
            Danh sách point_id (UUID string) đã được upsert
        """
        from qdrant_client.models import PointStruct

        if not chunks:
            logger.warning("Danh sách chunk rỗng, bỏ qua indexing.")
            return []

        doc_id = doc_id or str(uuid.uuid4())
        total = len(chunks)
        point_ids: list[str] = []

        logger.info(f"Indexing {total} chunks | doc_id={doc_id} | source={document.source}")

        # Tạo metadata template
        base_meta = dict(
            source=document.source,
            doc_id=doc_id,
            total_chunks=total,
            title=document.title,
            language=document.language,
            content_type=document.content_type,
            author=document.author,
            tags=document.tags,
        )

        # Xử lý theo batch
        for batch_start in tqdm(
            range(0, total, batch_size),
            desc="Indexing batches",
            unit="batch",
        ):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c.text for c in batch]

            # Gọi OpenAI Embedding API
            vectors = _embed_texts(texts)

            points: list[PointStruct] = []
            for i, (chunk, vector) in enumerate(zip(batch, vectors)):
                point_id = str(uuid.uuid4())
                meta = DocumentMetadata(
                    **base_meta,
                    chunk_index=chunk.chunk_index,
                )
                payload = meta.to_qdrant_payload()
                payload["text"] = chunk.text  # Lưu text gốc trong payload

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
                point_ids.append(point_id)

            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

        logger.success(f"Hoàn tất: {total} chunks → collection '{self.collection_name}'")
        return point_ids

    # ── Delete ─────────────────────────────────────────────────

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Xoá toàn bộ chunk thuộc một tài liệu theo doc_id."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        result = self._client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="doc_id",
                        match=MatchValue(value=doc_id),
                    )
                ]
            ),
        )
        logger.info(f"Đã xoá các chunk của doc_id={doc_id}")
        return result.status

    def collection_info(self) -> dict:
        """Trả về thông tin collection (số vectors, config…)."""
        info = self._client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "vectors_count": info.vectors_count,
            "status": str(info.status),
        }
