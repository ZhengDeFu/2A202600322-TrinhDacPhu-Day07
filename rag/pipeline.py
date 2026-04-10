"""
pipeline.py – Orchestrator kết hợp toàn bộ pipeline RAG.

RAGPipeline cung cấp 2 phương thức chính:
  - ingest(documents)          : Load → Chunk → Embed → Upsert vào Qdrant
  - query(question, ...)       : Embed query → Search → Generate → Return

Ví dụ sử dụng:
    from rag import RAGPipeline
    from rag.schema import DocumentInput

    pipeline = RAGPipeline()

    # Ingest
    doc = DocumentInput(content="...", source="https://...", title="...")
    pipeline.ingest([doc])

    # Query
    result = pipeline.query("Câu hỏi của bạn?")
    print(result.format())
"""

from __future__ import annotations

from loguru import logger

from rag.config import settings
from rag.schema.metadata import DocumentInput, RetrievedChunk
from rag.ingestion.loader import DocumentLoader
from rag.ingestion.chunker import TextChunker
from rag.ingestion.indexer import QdrantIndexer
from rag.retrieval.retriever import QdrantRetriever, MetadataFilter
from rag.generation.generator import OpenAIGenerator, GenerationResult


class RAGPipeline:
    """
    Pipeline RAG đầy đủ (Ingest + Retrieve + Generate).

    Args:
        collection_name    : Tên collection Qdrant (mặc định từ config)
        chunk_size         : Kích thước chunk (ký tự)
        chunk_overlap      : Overlap giữa các chunk (ký tự)
        top_k              : Số chunk trả về khi search
        score_threshold    : Ngưỡng score tối thiểu
        openai_temperature : Temperature của LLM
    """

    def __init__(
        self,
        collection_name: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        openai_temperature: float = 0.3,
    ) -> None:
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.indexer = QdrantIndexer(collection_name=collection_name)
        self.retriever = QdrantRetriever(collection_name=collection_name)
        self.generator = OpenAIGenerator(temperature=openai_temperature)

        self._top_k = top_k or settings.retrieval_top_k
        self._score_threshold = score_threshold or settings.retrieval_score_threshold

        logger.info("RAGPipeline khởi tạo thành công ✓")

    # ── Ingest ────────────────────────────────────────────────

    def ingest(
        self,
        documents: list[DocumentInput],
        batch_size: int = 64,
    ) -> dict[str, list[str]]:
        """
        Ingest danh sách tài liệu vào Qdrant.

        Args:
            documents  : Danh sách DocumentInput
            batch_size : Số chunk gửi Qdrant mỗi lần

        Returns:
            Dict {source: [point_ids]} ghi lại ID đã lưu
        """
        results: dict[str, list[str]] = {}

        for doc in documents:
            logger.info(f"Đang ingest: {doc.source}")
            chunks = self.chunker.split(doc.content)

            if not chunks:
                logger.warning(f"Không có chunk nào từ: {doc.source}")
                continue

            point_ids = self.indexer.index(
                document=doc,
                chunks=chunks,
                batch_size=batch_size,
            )
            results[doc.source] = point_ids

        total_points = sum(len(v) for v in results.values())
        logger.success(
            f"Ingest hoàn tất: {len(results)} tài liệu | {total_points} chunks"
        )
        return results

    def ingest_url(
        self,
        url: str,
        language: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Shortcut: load URL + ingest."""
        doc = self.loader.load_from_url(url, language=language, tags=tags)
        result = self.ingest([doc])
        return result.get(doc.source, [])

    def ingest_file(
        self,
        path: str,
        language: str | None = None,
        tags: list[str] | None = None,
    ) -> list[str]:
        """Shortcut: load file + ingest."""
        doc = self.loader.load_from_file(path, language=language, tags=tags)
        result = self.ingest([doc])
        return result.get(doc.source, [])

    # ── Query ─────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int | None = None,
        filters: MetadataFilter | None = None,
        score_threshold: float | None = None,
        stream: bool = False,
    ) -> GenerationResult:
        """
        Hỏi hệ thống RAG.

        Args:
            question       : Câu hỏi người dùng
            top_k          : Số chunk dùng làm context (mặc định từ config)
            filters        : Lọc theo metadata (tuỳ chọn)
            score_threshold: Ngưỡng score (tuỳ chọn)
            stream         : Nếu True, stream output từ LLM

        Returns:
            GenerationResult với answer + sources
        """
        logger.info(f"Query: {question[:80]}...")

        # 1. Retrieve
        chunks = self.retriever.search(
            query=question,
            top_k=top_k or self._top_k,
            filters=filters,
            score_threshold=score_threshold,
        )
        logger.debug(f"Đã retrieve {len(chunks)} chunks")

        # 2. Generate
        result = self.generator.generate(
            question=question,
            chunks=chunks,
            stream=stream,
        )

        return result

    # ── Utils ─────────────────────────────────────────────────

    def collection_info(self) -> dict:
        """Trả về thông tin collection Qdrant."""
        return self.indexer.collection_info()
