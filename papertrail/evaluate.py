"""Retrieval quality check: does search find the right paper?

We need questions whose correct paper is known, without labelling anything by hand, so we use the
papers themselves. A fixed random sample of papers (the same every run) is turned into questions:

  title     the paper's title. Easy for keyword search, because the words match exactly.
  sentence  one sentence from the middle of the abstract. Shares meaning and some words with the
            paper, but is not the title, so it is harder.

For each question we search and check where the source paper ranks. These are stand-ins for real
questions, so the numbers compare search methods and settings with each other; they do not predict
how often real questions are answered well.
"""

import re
import statistics
import time

from . import cache
from .retrieve import Retriever
from .timing import Timer


def _middle_sentence(abstract: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 40]
    return sentences[len(sentences) // 2] if sentences else abstract[:200]


def sample_questions(retriever: Retriever, kind: str, n: int) -> list[tuple[int, str]]:
    rows = retriever.db.execute(
        f"SELECT row_id, title, abstract FROM papers USING SAMPLE reservoir({n} ROWS) REPEATABLE (42)").fetchall()
    return [(row_id, title if kind == "title" else _middle_sentence(abstract)) for row_id, title, abstract in rows]


def evaluate(retriever: Retriever, kind: str, n: int, modes: list[str]) -> dict[str, dict]:
    """Returns {mode: {recall@1, recall@10, mrr, median_ms, p95_ms}} for each search mode."""
    was_enabled, cache.enabled = cache.enabled, False  # measure the search itself, not cached vectors
    try:
        questions = sample_questions(retriever, kind, n)
        # search() returns papers, not row numbers, so look up the arXiv id of each source paper
        ids = dict(retriever.db.execute(
            "SELECT row_id, id FROM papers WHERE row_id IN (SELECT unnest(?))", [[r for r, _ in questions]]).fetchall())
        results = {}
        for mode in modes:
            base, rerank = mode.removesuffix("+rerank"), mode.endswith("+rerank")  # e.g. "hybrid+rerank"
            retriever.search("warm up", 1, mode=base, rerank=rerank)
            ranks, times = [], []
            for row_id, text in questions:
                start = time.perf_counter()
                found = retriever.search(text, 10, None, Timer(), mode=base, rerank=rerank)
                times.append((time.perf_counter() - start) * 1000)
                found_ids = [p["id"] for p in found]
                ranks.append(found_ids.index(ids[row_id]) + 1 if ids[row_id] in found_ids else None)
            total = len(ranks)
            times.sort()
            results[mode] = {
                "recall@1": sum(r == 1 for r in ranks) / total,
                "recall@10": sum(r is not None for r in ranks) / total,
                "mrr": sum(1 / r for r in ranks if r) / total,
                "median_ms": statistics.median(times),
                "p95_ms": times[int(0.95 * total) - 1],
            }
        return results
    finally:
        cache.enabled = was_enabled
