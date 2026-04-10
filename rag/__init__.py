"""
RAG System with Qdrant
======================
Hệ thống Retrieval-Augmented Generation sử dụng:
  - Qdrant       : Vector store
  - sentence-transformers : Embedding local
  - Google Gemini: LLM generation
"""

from rag.pipeline import RAGPipeline
from rag.schema.metadata import DocumentMetadata
from rag.config import settings

__version__ = "0.1.0"
__all__ = ["RAGPipeline", "DocumentMetadata", "settings"]
