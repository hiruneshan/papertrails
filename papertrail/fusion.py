"""Reciprocal Rank Fusion: merge several ranked lists into one.

Each list says "paper A is 1st, paper B is 2nd, ...". A paper earns 1 / (RRF_K + position)
from every list it appears in, and the earned points are added up. Only positions are used,
never the scores, because keyword (BM25) and vector (cosine) scores are on different scales.
A paper that is high in both lists beats one that is first in just one.
"""

RRF_K = 60  # the standard constant from the original paper (Cormack et al., 2009); damps the lead of the very top ranks


def rrf(*ranked_lists: list[int], k: int = RRF_K) -> list[tuple[int, float]]:
    """Return [(item, fused_score), ...] best first. Each input list is ordered best first."""
    scores: dict[int, float] = {}
    for items in ranked_lists:
        for position, item in enumerate(items, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + position)
    # ties (same fused score) keep the order of first appearance, so vector results win ties
    return sorted(scores.items(), key=lambda kv: -kv[1])
