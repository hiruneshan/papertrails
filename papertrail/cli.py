"""Command line: papertrail build-index | search | ask | serve"""

import typer

from . import config

app = typer.Typer(help="PaperTrail: ask questions about arXiv computer science papers.", no_args_is_help=True)


def _print_papers(papers: list[dict]) -> None:
    for i, p in enumerate(papers, 1):
        print(f"{i}. [{p['score']:.3f}] {p['title']}")
        print(f"   {p['url']}  ({p['categories']}, {p['update_date']})")


@app.command("build-index")
def build_index():
    """Build the FAISS index and DuckDB table from vectors.npy + papers.parquet."""
    from .index import build
    build()


@app.command()
def search(
    question: str,
    k: int = typer.Option(config.TOP_K, help="How many papers to return"),
    category: str = typer.Option(None, help="Only this category, e.g. cs.LG"),
):
    """Find the papers closest to a question (no LLM call)."""
    from .retrieve import Retriever
    _print_papers(Retriever().search(question, k, category))


@app.command()
def ask(
    question: str,
    k: int = typer.Option(config.TOP_K, help="How many papers to give Claude"),
    category: str = typer.Option(None, help="Only this category, e.g. cs.LG"),
):
    """Answer a question from the most relevant papers, with citations."""
    from .generate import answer
    from .retrieve import Retriever
    papers = Retriever().search(question, k, category)
    print(answer(question, papers))
    print("\nSources:")
    _print_papers(papers)


@app.command()
def serve(port: int = 8000):
    """Start the API at http://localhost:PORT (docs at /docs)."""
    import uvicorn
    uvicorn.run("papertrail.api:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
