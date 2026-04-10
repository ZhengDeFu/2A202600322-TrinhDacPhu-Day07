"""
scripts/query.py – CLI để hỏi hệ thống RAG.

Sử dụng:
    python scripts/query.py ask "VinFast VF8 có tầm vận hành bao nhiêu?"
    python scripts/query.py ask "..." --top-k 3 --language vi --stream
    python scripts/query.py ask "..." --source "https://vinfastanthai.com/..."
    python scripts/query.py interactive
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint
from loguru import logger

from rag.pipeline import RAGPipeline
from rag.retrieval.retriever import MetadataFilter

app = typer.Typer(
    name="query",
    help="🔍 Hỏi hệ thống RAG",
    no_args_is_help=True,
)
console = Console()


def _make_filter(
    source: str | None,
    language: str | None,
    content_type: str | None,
    tags: list[str],
    doc_id: str | None,
) -> MetadataFilter | None:
    """Tạo MetadataFilter từ các tham số CLI."""
    f = MetadataFilter(
        source=source,
        language=language,
        content_type=content_type,
        tags=tags,
        doc_id=doc_id,
    )
    # Chỉ trả về filter nếu có ít nhất một điều kiện
    has_condition = any([source, language, content_type, tags, doc_id])
    return f if has_condition else None


# ── Subcommand: ask ───────────────────────────────────────────

@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Câu hỏi cần trả lời"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Số chunk context tối đa"),
    score_threshold: float = typer.Option(0.3, "--threshold", help="Ngưỡng score tối thiểu"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output từ LLM"),
    show_chunks: bool = typer.Option(False, "--show-chunks", "-c", help="Hiển thị các chunk retrieved"),
    # Metadata filters
    source: str = typer.Option(None, "--source", help="Lọc theo source URL/path"),
    language: str = typer.Option(None, "--language", "-l", help="Lọc theo ngôn ngữ"),
    content_type: str = typer.Option(None, "--type", help="Lọc theo loại nội dung"),
    tags: list[str] = typer.Option([], "--tags", "-t", help="Lọc theo tags"),
    doc_id: str = typer.Option(None, "--doc-id", help="Lọc theo doc_id"),
):
    """Đặt một câu hỏi cho hệ thống RAG."""
    console.print(f"\n[bold yellow]❓ Câu hỏi:[/bold yellow] {question}\n")

    pipeline = RAGPipeline()
    filters = _make_filter(source, language, content_type, tags, doc_id)

    result = pipeline.query(
        question=question,
        top_k=top_k,
        filters=filters,
        score_threshold=score_threshold,
        stream=stream,
    )

    # Hiển thị chunks nếu yêu cầu
    if show_chunks and result.chunks:
        table = Table(
            title=f"📦 Retrieved Chunks ({len(result.chunks)})",
            show_header=True,
            header_style="bold blue",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", style="cyan", width=6)
        table.add_column("Source", style="yellow", max_width=40)
        table.add_column("Preview", style="white", max_width=60)

        for i, chunk in enumerate(result.chunks, 1):
            table.add_row(
                str(i),
                f"{chunk.score:.3f}",
                chunk.metadata.source[:40],
                chunk.text[:100].replace("\n", " ") + "...",
            )
        console.print(table)
        console.print()

    # Hiển thị câu trả lời
    console.print(
        Panel(
            Markdown(result.answer),
            title="[bold green]💬 Câu trả lời",
            border_style="green",
        )
    )

    # Hiển thị sources
    if result.sources:
        console.print("\n[bold]📚 Nguồn tham khảo:[/bold]")
        for i, src in enumerate(result.sources, 1):
            console.print(f"  [dim][{i}][/dim] [blue]{src}[/blue]")
    console.print()


# ── Subcommand: interactive ───────────────────────────────────

@app.command("interactive")
def interactive(
    top_k: int = typer.Option(5, "--top-k", "-k"),
    language: str = typer.Option(None, "--language", "-l"),
    show_chunks: bool = typer.Option(False, "--show-chunks", "-c"),
):
    """Chế độ hỏi đáp liên tục (REPL)."""
    console.print(
        Panel(
            "[bold cyan]🤖 RAG Interactive Mode[/bold cyan]\n"
            "Gõ câu hỏi và nhấn Enter. Gõ [red]exit[/red] hoặc [red]quit[/red] để thoát.",
            border_style="cyan",
        )
    )

    pipeline = RAGPipeline()
    filters = MetadataFilter(language=language) if language else None

    while True:
        try:
            question = console.input("\n[bold yellow]❓ Câu hỏi > [/bold yellow]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Tạm biệt![/dim]")
            break

        if question.lower() in {"exit", "quit", "q", "thoát"}:
            console.print("\n[dim]Tạm biệt![/dim]")
            break

        if not question:
            continue

        result = pipeline.query(
            question=question,
            top_k=top_k,
            filters=filters,
        )

        if show_chunks and result.chunks:
            for i, chunk in enumerate(result.chunks, 1):
                console.print(
                    f"  [dim][{i}] score={chunk.score:.2f} | {chunk.metadata.source[:50]}[/dim]"
                )

        console.print(
            Panel(
                Markdown(result.answer),
                title="[bold green]💬 Trả lời",
                border_style="green",
                padding=(1, 2),
            )
        )

        if result.sources:
            console.print("[dim]Nguồn: " + " | ".join(result.sources[:3]) + "[/dim]")


if __name__ == "__main__":
    app()
