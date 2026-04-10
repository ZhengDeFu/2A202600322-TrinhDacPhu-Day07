"""
generator.py – Tạo câu trả lời từ LLM dựa trên context đã retrieved.

Sử dụng OpenAI thay cho Gemini (Mặc định: gpt-4o-mini).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from rag.config import settings
from rag.schema.metadata import RetrievedChunk


# ─────────────────────────────────────────────────────────────────────────────
# GenerationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Kết quả sinh câu trả lời từ LLM."""
    answer: str
    sources: list[str] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    model: str = ""

    def format(self) -> str:
        """Định dạng kết quả để in ra terminal."""
        lines = [
            "─" * 60,
            "📝 Câu trả lời:",
            self.answer,
            "",
            "📚 Nguồn tham khảo:",
        ]
        for i, src in enumerate(self.sources, 1):
            lines.append(f"  [{i}] {src}")
        lines.append("─" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

Nguyên tắc:
1. Chỉ sử dụng thông tin từ ngữ cảnh (CONTEXT) bên dưới để trả lời.
2. Nếu không đủ thông tin để trả lời, hãy nói rõ "Tôi không tìm thấy thông tin liên quan trong tài liệu."
3. Trích dẫn số thứ tự nguồn [1], [2], ... khi sử dụng thông tin từ đó.
4. Trả lời ngắn gọn, súc tích và chính xác.
5. Ngôn ngữ trả lời phải phù hợp với ngôn ngữ câu hỏi."""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Tạo block context từ danh sách chunk."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.source
        title = chunk.metadata.title or ""
        score = f"{chunk.score:.2f}"
        header = f"[{i}] Nguồn: {source}"
        if title:
            header += f" | Tiêu đề: {title}"
        header += f" | Score: {score}"
        lines.append(header)
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIGenerator
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIGenerator:
    """
    Generator sử dụng OpenAI Chat API (gpt-4o-mini).
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Cài đặt `openai` để dùng OpenAIGenerator"
            ) from e

        self.model_name = model or settings.openai_chat_model
        api_key = api_key or settings.openai_api_key

        if not api_key:
            raise ValueError("OPENAI_API_KEY chưa được cấu hình.")

        self._client = OpenAI(api_key=api_key)
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.debug(f"OpenAIGenerator sẵn sàng: model={self.model_name}")

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        stream: bool = False,
    ) -> GenerationResult:
        """
        Sinh câu trả lời từ câu hỏi và danh sách chunk retrieved.
        """
        if not chunks:
            return GenerationResult(
                answer="Tôi không tìm thấy tài liệu nào liên quan trong kho dữ liệu.",
                sources=[],
                chunks=[],
                model=self.model_name,
            )

        context_text = _build_context_block(chunks)
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"--- CONTEXT ---\n{context_text}\n\n--- CÂU HỎI ---\n{question}"}
        ]

        logger.debug(f"Calling OpenAI | model={self.model_name}")

        try:
            if stream:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=True,
                )
                answer_parts = []
                for chunk in response:
                    content = chunk.choices[0].delta.content or ""
                    print(content, end="", flush=True)
                    answer_parts.append(content)
                print()
                answer = "".join(answer_parts)
            else:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                answer = response.choices[0].message.content

        except Exception as e:
            logger.error(f"Lỗi khi gọi OpenAI: {e}")
            raise

        # Deduplicate sources
        sources: list[str] = []
        seen: set[str] = set()
        for chunk in chunks:
            src = chunk.metadata.source
            if src not in seen:
                sources.append(src)
                seen.add(src)

        return GenerationResult(
            answer=answer.strip(),
            sources=sources,
            chunks=chunks,
            model=self.model_name,
        )
