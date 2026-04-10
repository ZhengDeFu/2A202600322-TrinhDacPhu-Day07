"""
scripts/ingest.py – CLI để đưa tài liệu vào Qdrant.

Sử dụng:
    python scripts/ingest.py url  "https://example.com/article" --language vi
    python scripts/ingest.py file "./data/report.pdf" --tags vinfast --tags xe-dien
    python scripts/ingest.py info
"""

from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo import được package rag từ thư mục gốc
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from rag.pipeline import RAGPipeline
from rag.retrieval.retriever import MetadataFilter

app = typer.Typer(
    name="ingest",
    help="📥 Đưa tài liệu vào hệ thống RAG (Qdrant)",
    no_args_is_help=True,
)
console = Console()


def _get_pipeline() -> RAGPipeline:
    return RAGPipeline()


# ── Subcommand: url ───────────────────────────────────────────

@app.command("url")
def ingest_url(
    url: str = typer.Argument(..., help="URL cần ingest"),
    language: str = typer.Option(None, "--language", "-l", help="Mã ngôn ngữ (vi, en, ...)"),
    tags: list[str] = typer.Option([], "--tags", "-t", help="Nhãn phân loại (có thể lặp)"),
):
    """Fetch và ingest một trang web từ URL."""
    console.print(f"\n[bold cyan]🌐 Đang fetch URL:[/bold cyan] {url}")
    pipeline = _get_pipeline()
    point_ids = pipeline.ingest_url(url, language=language or None, tags=tags)
    _print_success(point_ids, source=url)


# ── Subcommand: file ──────────────────────────────────────────

@app.command("file")
def ingest_file(
    path: str = typer.Argument(..., help="Đường dẫn tệp (.pdf, .txt, .md, .docx)"),
    language: str = typer.Option(None, "--language", "-l", help="Mã ngôn ngữ"),
    tags: list[str] = typer.Option([], "--tags", "-t", help="Nhãn phân loại"),
    author: str = typer.Option(None, "--author", "-a", help="Tác giả"),
):
    """Ingest một tệp tài liệu (PDF, DOCX, TXT, Markdown)."""
    from rag.ingestion.loader import DocumentLoader

    p = Path(path)
    console.print(f"\n[bold cyan]📄 Đang load file:[/bold cyan] {p.resolve()}")

    loader = DocumentLoader()
    doc = loader.load_from_file(
        path=p,
        language=language or None,
        author=author or None,
        tags=tags,
    )

    pipeline = _get_pipeline()
    result = pipeline.ingest([doc])
    point_ids = result.get(doc.source, [])
    _print_success(point_ids, source=doc.source)


# ── Subcommand: info ──────────────────────────────────────────

@app.command("info")
def collection_info():
    """Hiển thị thông tin collection Qdrant."""
    pipeline = _get_pipeline()
    info = pipeline.collection_info()

    table = Table(title="📊 Qdrant Collection Info", show_header=True, header_style="bold magenta")
    table.add_column("Thuộc tính", style="cyan")
    table.add_column("Giá trị", style="green")

    for k, v in info.items():
        table.add_row(str(k), str(v))

    console.print()
    console.print(table)
    console.print()


# ── Helpers ───────────────────────────────────────────────────

def _print_success(point_ids: list[str], source: str) -> None:
    if point_ids:
        console.print(
            Panel(
                f"[green]✅ Ingest thành công![/green]\n"
                f"Source : [bold]{source}[/bold]\n"
                f"Chunks : [bold]{len(point_ids)}[/bold]\n"
                f"IDs    : {point_ids[0]}... (+{len(point_ids)-1} more)",
                title="[bold green]Kết quả",
                border_style="green",
            )
        )
    else:
        console.print("[yellow]⚠️  Không có chunk nào được ingest.[/yellow]")


if __name__ == "__main__":
    app()
