"""Timing and run logs.

Every run (search, ask, bench, build-index, API request) is appended to:
  logs/runs.jsonl   - one JSON record per run, for `papertrail report`
  logs/timings.log  - the same, one readable line per run
Changes you make are recorded in logs/CHANGES.md by `papertrail log-change`,
and every later run is tagged with that change's label.
"""

import json
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import config

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
RUNS = LOG_DIR / "runs.jsonl"
READABLE = LOG_DIR / "timings.log"
CHANGES = LOG_DIR / "CHANGES.md"
LABEL_FILE = LOG_DIR / ".current_label"

_lock = threading.Lock()

# Stages in pipeline order (used for display).
STAGE_ORDER = [
    "import_libs", "load_model", "load_index", "open_db",                       # startup
    "qvec_cache", "embed_query", "vector_search", "db_lookup",                  # per question: search
    "answer_cache", "answer_wait", "llm",                                       # per question: answer
    "load_vectors", "write_db", "build_hnsw", "save_index",                     # build-index
]


class Timer:
    """Collects how long each named stage takes."""

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.info: dict = {}
        self._start = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t

    def total(self) -> float:
        return time.perf_counter() - self._start


def current_label() -> str:
    try:
        return LABEL_FILE.read_text().strip() or "unlabelled"
    except FileNotFoundError:
        return "unlabelled"


def git_commit() -> str:
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (commit or "no-commits") + ("+changes" if dirty else "")
    except Exception:
        return "unknown"


def settings() -> dict:
    """The knobs that affect speed or quality, saved with every run."""
    return {
        "provider": config.LLM_PROVIDER,
        "llm_model": config.OPENROUTER_MODEL if config.LLM_PROVIDER == "openrouter" else config.CLAUDE_MODEL,
        "embed_model": config.EMBED_MODEL,
        "embed_backend": config.EMBED_BACKEND,
        "encoder_threads": config.ENCODER_THREADS,
        "top_k": config.TOP_K,
        "hnsw_m": config.HNSW_M,
        "hnsw_ef_construction": config.HNSW_EF_CONSTRUCTION,
        "hnsw_ef_search": config.HNSW_EF_SEARCH,
        "cpu_threads": config.CPU_THREADS,
        "cache": config.CACHE,
        "prompt_version": config.PROMPT_VERSION,
    }


def fmt(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def log_run(command: str, timer: Timer, total: float | None = None, **extra) -> dict:
    total = timer.total() if total is None else total
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "label": current_label(),
        "command": command,
        "commit": git_commit(),
        "total_ms": round(total * 1000, 1),
        "stages_ms": {k: round(v * 1000, 1) for k, v in timer.stages.items()},
        **timer.info,
        **extra,
        "settings": settings(),
    }
    ordered = sorted(timer.stages, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)
    stages = ", ".join(f"{s} {fmt(timer.stages[s])}" for s in ordered)
    line = f"{record['time'].replace('T', ' ')} | {record['label']} | {command} | total {fmt(total)} | {stages}"
    if "question" in extra:
        line += f' | "{extra["question"][:70]}"'
    LOG_DIR.mkdir(exist_ok=True)
    with _lock:
        with RUNS.open("a") as f:
            f.write(json.dumps(record) + "\n")
        with READABLE.open("a") as f:
            f.write(line + "\n")
    return record


def log_change(label: str, description: str) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    if not CHANGES.exists():
        CHANGES.write_text("# PaperTrail change log\n\nEach change gets a label; every run after it is tagged with that label "
                           "in runs.jsonl and timings.log. Compare them with `papertrail report`.\n")
    s = settings()
    with CHANGES.open("a") as f:
        f.write(f"\n## {label}\n\n- **When:** {datetime.now():%Y-%m-%d %H:%M}\n- **Commit:** {git_commit()}\n"
                f"- **What changed:** {description}\n"
                f"- **Settings:** {', '.join(f'{k}={v}' for k, v in s.items())}\n")
    LABEL_FILE.write_text(label)


def load_runs() -> list[dict]:
    if not RUNS.exists():
        return []
    return [json.loads(line) for line in RUNS.read_text().splitlines() if line.strip()]
