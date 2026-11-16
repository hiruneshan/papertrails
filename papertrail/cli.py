"""Command line: papertrail build-index | search | ask | serve | bench | report | log-change"""

import statistics
from pathlib import Path

import typer

from . import cache, config
from .timing import STAGE_ORDER, Timer, current_label, fmt, load_runs, log_change, log_run

app = typer.Typer(help="PaperTrail: ask questions about arXiv computer science papers.", no_args_is_help=True)

BENCH_FILE = Path(__file__).resolve().parent.parent / "bench" / "questions.txt"


def _print_papers(papers: list[dict]) -> None:
    for i, p in enumerate(papers, 1):
        print(f"{i}. [{p['score']:.3f}] {p['title']}")
        print(f"   {p['url']}  ({p['categories']}, {p['update_date']})")


def _print_timing(timer: Timer) -> None:
    ordered = sorted(timer.stages, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
    print("\nTiming: " + ", ".join(f"{s} {fmt(timer.stages[s])}" for s in ordered) + f" | total {fmt(timer.total())}")


def _load_retriever(timer: Timer):
    with timer.stage("import_libs"):
        from .retrieve import Retriever
    return Retriever(timer)


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
    timer = Timer()
    papers = _load_retriever(timer).search(question, k, category, timer)
    _print_papers(papers)
    _print_timing(timer)
    log_run("search", timer, question=question, k=k, category=category)


@app.command()
def ask(
    question: str,
    k: int = typer.Option(config.TOP_K, help="How many papers to give the LLM"),
    category: str = typer.Option(None, help="Only this category, e.g. cs.LG"),
):
    """Answer a question from the most relevant papers, with citations."""
    timer = Timer()
    retriever = _load_retriever(timer)
    from .generate import make_client, model_name
    from .pipeline import ask as run_ask
    print(f"Answering with {config.LLM_PROVIDER} / {model_name()} ...\n")
    text, papers = run_ask(retriever, make_client(), question, k, category, timer)
    if timer.info.get("answer_cache_hit"):
        print(f"(cached answer from {timer.info['answer_cached_at']}; same papers as then)\n")
    print(text)
    print("\nSources:")
    _print_papers(papers)
    _print_timing(timer)
    log_run("ask", timer, question=question, k=k, category=category)


@app.command()
def bench(
    repeat: int = typer.Option(3, help="Times to run each question (the median is reported)"),
    with_llm: bool = typer.Option(False, "--with-llm", help="Also call the LLM (uses your free daily quota)"),
    use_cache: bool = typer.Option(False, "--cache", help="Use the Redis cache (off by default, to time the raw pipeline)"),
):
    """Time the fixed question set in bench/questions.txt and log the results."""
    cache.enabled = use_cache and config.CACHE == "redis"
    if use_cache and not cache.available():
        raise SystemExit(f"--cache needs Redis at {config.REDIS_URL}, and it isn't reachable.")
    questions = [q.strip() for q in BENCH_FILE.read_text().splitlines() if q.strip()]
    startup = Timer()
    retriever = _load_retriever(startup)
    log_run("bench-startup", startup)
    print(f"startup: " + ", ".join(f"{s} {fmt(v)}" for s, v in startup.stages.items()))

    if with_llm:
        from .generate import make_client
        from .pipeline import ask as run_ask
        client = make_client()

    retriever.search("warm up", 1)  # first search pays one-off costs; don't count it
    per_stage: dict[str, list[float]] = {}
    totals = []
    for r in range(repeat):
        for q in questions:
            t = Timer()
            if with_llm and (r == 0 or use_cache):  # without the cache, call the LLM once per question to save quota
                run_ask(retriever, client, q, config.TOP_K, None, t)
            else:
                retriever.search(q, config.TOP_K, None, t)
            total = t.total()
            totals.append(total)
            for s, v in t.stages.items():
                per_stage.setdefault(s, []).append(v)
            log_run("bench-query", t, total=total, question=q, repeat=r, cache=cache.enabled)

    print(f"\n{len(questions)} questions x {repeat} runs (median / slowest):")
    for s in sorted(per_stage, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99):
        v = per_stage[s]
        print(f"  {s:14s} {fmt(statistics.median(v)):>8s} / {fmt(max(v)):>8s}")
    print(f"  {'total':14s} {fmt(statistics.median(totals)):>8s} / {fmt(max(totals)):>8s}")
    print(f"\nlogged under label '{current_label()}' - compare with `papertrail report`")


@app.command()
def report(command: str = typer.Option(None, help="Only this command, e.g. bench-query or ask")):
    """Compare timings across labels (changes): median milliseconds per stage."""
    runs = [r for r in load_runs() if not command or r["command"] == command]
    if not runs:
        print("No runs logged yet.")
        return
    groups: dict[tuple, list[dict]] = {}
    for r in runs:
        cmd = r["command"]
        if "answer_cache_hit" in r:
            cmd += " hit" if r["answer_cache_hit"] else " miss"
        elif r.get("qvec_cache_hit"):
            cmd += " vec-hit"
        groups.setdefault((r["label"], cmd), []).append(r)
    stages = [s for s in STAGE_ORDER if any(s in r["stages_ms"] for r in runs)]
    short = {"import_libs": "import", "load_model": "model", "load_index": "index", "open_db": "db",
             "qvec_cache": "qcache", "embed_query": "embed", "vector_search": "search", "db_lookup": "lookup",
             "answer_cache": "acache", "answer_wait": "await", "llm": "llm",
             "load_vectors": "loadvec", "write_db": "writedb", "build_hnsw": "hnsw", "save_index": "save"}
    header = f"{'label':22s} {'command':16s} {'runs':>4s} " + " ".join(f"{short.get(s, s)[:8]:>8s}" for s in stages) + f" {'total':>9s}"
    print("median time per stage (ms)\n" + header + "\n" + "-" * len(header))
    for (label, cmd), rs in groups.items():
        cells = []
        for s in stages:
            vals = [r["stages_ms"][s] for r in rs if s in r["stages_ms"]]
            cells.append(f"{statistics.median(vals):8.0f}" if vals else f"{'-':>8s}")
        total = statistics.median(r["total_ms"] for r in rs)
        print(f"{label[:22]:22s} {cmd[:16]:16s} {len(rs):4d} " + " ".join(cells) + f" {total:9.0f}")


@app.command("log-change")
def log_change_cmd(
    label: str = typer.Argument(..., help="Short name for this change, e.g. v1-onnx-int8"),
    description: str = typer.Argument(..., help="What you changed and why"),
):
    """Record a change in logs/CHANGES.md; later runs are tagged with its label."""
    log_change(label, description)
    print(f"Logged change '{label}'. Runs from now on are tagged '{label}'.\n"
          f"Next: `papertrail bench` to measure it, then `papertrail report` to compare.")


@app.command("cache-stats")
def cache_stats():
    """Show what the Redis cache holds."""
    for k, v in cache.stats().items():
        print(f"{k:30s} {v}")


@app.command("cache-clear")
def cache_clear():
    """Delete all PaperTrail entries from Redis."""
    print(f"deleted {cache.clear()} keys")


@app.command()
def serve(port: int = 8000):
    """Start the API at http://localhost:PORT (docs at /docs)."""
    import uvicorn
    uvicorn.run("papertrail.api:app", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
