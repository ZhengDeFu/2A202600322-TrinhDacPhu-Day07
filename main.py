"""
main.py – FastAPI Server & CLI Manual Demo cho hệ thống RAG (Hybrid Mode).

Cách dùng:
  1. Chạy Web Server:  uvicorn main:app --reload
  2. Chạy Manual Demo: python3 main.py [câu hỏi]
"""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

# Import các thành phần RAG đã xây dựng
from rag.config import settings
from rag.schema.metadata import DocumentInput
from rag.ingestion.loader import DocumentLoader
from rag.pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG & SAMPLES
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

SAMPLE_FILES = [
    "data/python_intro.txt",
    "data/vector_store_notes.md",
    "data/rag_system_design.md",
]

# ─────────────────────────────────────────────────────────────────────────────
# Global RAG Pipeline (Singleton)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None

def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CLI HELPERS (Theo mẫu yêu cầu)
# ─────────────────────────────────────────────────────────────────────────────

def load_documents_from_files(file_paths: list[str]) -> list[DocumentInput]:
    """Sử dụng DocumentLoader hiện có để đọc danh sách file."""
    loader = DocumentLoader()
    documents: list[DocumentInput] = []
    
    for raw_path in file_paths:
        path = Path(raw_path)
        if not path.exists():
            logger.warning(f"Bỏ qua file không tồn tại: {path}")
            continue
            
        try:
            # Tự động nhận diện định dạng trong load_from_file
            doc = loader.load_from_file(path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"Lỗi khi load {path}: {e}")
            
    return documents

def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    """Chế độ CLI Demo: Nạp tài liệu mẫu và trả lời câu hỏi."""
    files = sample_files or SAMPLE_FILES
    query = question or "Tóm tắt các thông tin chính từ các tài liệu đã nạp."

    print("\n" + "═"*50)
    print("🚀 ĐANG CHẠY RAG MANUAL DEMO (CLI MODE)")
    print("═"*50)
    
    # 1. Load files
    print(f"\n[1/3] Đang đọc {len(files)} file mẫu...")
    docs = load_documents_from_files(files)
    if not docs:
        print("❌ Không có file nào được nạp thành công. Hãy tạo thư mục 'data/' và thêm file mẫu.")
        return 1

    # 2. Ingest vào Vector DB (Qdrant)
    print(f"\n[2/3] Đang nạp {len(docs)} tài liệu vào Qdrant (OpenAI Embedding)...")
    pipeline = get_pipeline()
    pipeline.ingest(docs)

    # 3. Query & Answer
    print(f"\n[3/3] Đang thực hiện RAG cho câu hỏi: '{query}'")
    result = pipeline.query(query)

    print("\n" + "✨ CÂU TRẢ LỜI TỪ AGENT:")
    print("━"*50)
    print(result.answer)
    print("━"*50)
    
    print("\n📚 Nguồn tham khảo:")
    for source in result.sources:
        print(f"  - {source}")
    
    return 0

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo pipeline khi server bắt đầu."""
    logger.info("Initializing RAG Pipeline for API...")
    get_pipeline()
    yield
    logger.info("Shutting down RAG Server...")

app = FastAPI(title="RAG System API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/query")
async def api_query(req: QueryRequest):
    pipeline = get_pipeline()
    result = pipeline.query(req.question, top_k=req.top_k)
    return {
        "answer": result.answer,
        "sources": result.sources,
        "chunks": [{"text": c.text[:200], "score": c.score} for c in result.chunks]
    }

@app.post("/ingest/url")
async def api_ingest_url(url: str, tags: list[str] = []):
    pipeline = get_pipeline()
    point_ids = pipeline.ingest_url(url, tags=tags)
    return {"status": "success", "point_ids": point_ids}

# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    # Nếu có tham số dòng lệnh hoặc không có uvicorn (chạy trực tiếp)
    # thì ưu tiên chạy Demo CLI
    if len(sys.argv) > 1 or not os.getenv("RUN_AS_SERVER"):
        question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
        return run_manual_demo(question=question)
    return 0

if __name__ == "__main__":
    # Lưu ý: Khi chạy uvicorn main:app, khối này sẽ không được thực thi.
    # Nó chỉ chạy khi bạn gõ: python3 main.py
    sys.exit(main())
