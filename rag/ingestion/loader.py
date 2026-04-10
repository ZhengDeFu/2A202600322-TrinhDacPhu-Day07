"""
loader.py – Load tài liệu từ nhiều nguồn khác nhau.

Hỗ trợ:
  - URL     : fetch HTML, parse title + body text (httpx + BeautifulSoup)
  - PDF     : extract text qua PyMuPDF (fitz)
  - DOCX    : extract text qua python-docx
  - Markdown / Plain text : đọc trực tiếp
  - Raw text: truyền thẳng string vào

Trả về DocumentInput để chuyển sang pipeline ingestion.
"""

from __future__ import annotations

import re
from pathlib import Path

from loguru import logger

from rag.schema.metadata import DocumentInput


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Loại bỏ các ký tự trắng dư thừa."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# DocumentLoader
# ─────────────────────────────────────────────────────────────────────────────

class DocumentLoader:
    """
    Loader đa nguồn. Tất cả phương thức đều trả về `DocumentInput`.

    Ví dụ:
        loader = DocumentLoader()
        doc = loader.load_from_url("https://example.com/article")
        doc = loader.load_from_file("/data/report.pdf")
        doc = loader.load_from_text("Nội dung...", source="internal")
    """

    # ── URL ───────────────────────────────────────────────────

    def load_from_url(
        self,
        url: str,
        language: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentInput:
        """Fetch URL và parse thành markdown dùng Firecrawl (self-hosted)."""
        try:
            from firecrawl import FirecrawlApp
        except ImportError as e:
            raise ImportError(
                "Cài đặt `firecrawl-py` để dùng load_from_url với Firecrawl"
            ) from e

        from rag.config import settings

        logger.info(f"Scraping URL via Firecrawl: {url}")
        
        app = FirecrawlApp(
            api_key=settings.firecrawl_api_key,
            api_url=settings.firecrawl_api_url
        )
        
        # Scrape URL sang markdown
        try:
            scrape_result = app.scrape(url, formats=['markdown'])
            try:
                # Firecrawl v1+ trả về Pydantic model (Document)
                res_dict = scrape_result.model_dump()
            except AttributeError:
                res_dict = scrape_result if isinstance(scrape_result, dict) else {}

            content = res_dict.get("markdown", "")
            title = res_dict.get("metadata", {}).get("title", url)
            
            if not content:
                logger.warning(f"Firecrawl returned empty content for {url}. Dùng fallback...")
                raise ValueError("Empty content from Firecrawl")
                
        except Exception as e:
            logger.warning(f"Firecrawl scrape failed for {url} ({e}). Đang dùng Fallback (httpx+bs4)...")
            content, title = self._fallback_scrape(url)



        content = _clean_text(content)

        return DocumentInput(
            content=content,
            source=url,
            title=title,
            language=language,
            content_type="web",
            tags=tags or [],
        )

    def _fallback_scrape(self, url: str) -> tuple[str, str | None]:
        """Dùng httpx + BeautifulSoup để cào dữ liệu tĩnh cơ bản nếu Firecrawl lỗi."""
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError as e:
            logger.error("Vui lòng cài đặt httpx và beautifulsoup4 (pip install httpx beautifulsoup4) để dùng fallback")
            raise

        with httpx.Client(follow_redirects=True, timeout=15) as client:
            # Fake User-Agent để tránh filter cấm bot cơ bản
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            response = client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        body = soup.get_text(separator="\n")
        return body, title

    # ── File (PDF / DOCX / Text / Markdown) ───────────────────

    def load_from_file(
        self,
        path: str | Path,
        language: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentInput:
        """Load tài liệu từ file. Hỗ trợ .pdf, .docx, .txt, .md."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        suffix = path.suffix.lower()
        logger.info(f"Loading file: {path} ({suffix})")

        if suffix == ".pdf":
            content, title = self._load_pdf(path)
            content_type = "pdf"
        elif suffix in {".docx", ".doc"}:
            content, title = self._load_docx(path)
            content_type = "docx"
        elif suffix == ".md":
            content = _clean_text(path.read_text(encoding="utf-8"))
            title = path.stem
            content_type = "markdown"
        else:  # .txt hoặc mặc định
            content = _clean_text(path.read_text(encoding="utf-8"))
            title = path.stem
            content_type = "text"

        return DocumentInput(
            content=content,
            source=str(path.resolve()),
            title=title,
            language=language,
            content_type=content_type,
            author=author,
            tags=tags or [],
        )

    def _load_pdf(self, path: Path) -> tuple[str, str | None]:
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError("Cài đặt `pymupdf` để load file PDF") from e

        doc = fitz.open(str(path))
        pages: list[str] = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()

        title = path.stem
        # Thử lấy metadata từ PDF
        meta = fitz.open(str(path)).metadata
        if meta and meta.get("title"):
            title = meta["title"]

        return _clean_text("\n".join(pages)), title

    def _load_docx(self, path: Path) -> tuple[str, str | None]:
        try:
            from docx import Document
        except ImportError as e:
            raise ImportError("Cài đặt `python-docx` để load file DOCX") from e

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n".join(paragraphs)), path.stem

    # ── Raw text ──────────────────────────────────────────────

    def load_from_text(
        self,
        text: str,
        source: str,
        title: str | None = None,
        language: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
    ) -> DocumentInput:
        """Load từ chuỗi văn bản trực tiếp."""
        logger.debug(f"Loading raw text from source: {source}")
        return DocumentInput(
            content=_clean_text(text),
            source=source,
            title=title,
            language=language,
            content_type="text",
            author=author,
            tags=tags or [],
        )
