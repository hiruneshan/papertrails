"""Cross-encoder reranker. Needs the exported model; tests skip if it is missing."""

import numpy as np
import pytest

from papertrail import config
from papertrail.reranker import Reranker

pytestmark = pytest.mark.skipif(not (config.RERANKER_DIR / config.RERANK_MODEL / "tokenizer.json").exists(),
                                reason="reranker not exported; run scripts/export_reranker.py")

QUESTION = "How can we defend language models against prompt injection?"
TEXTS = [
    "A recipe for sourdough bread with a long cold fermentation.",
    "We propose a defense that detects and blocks injected instructions before the language model follows them.",
    "We study the orbital mechanics of exoplanets in binary star systems.",
]


@pytest.fixture(scope="module")
def reranker():
    return Reranker()


def test_relevant_text_scores_highest(reranker):
    scores = reranker.score(QUESTION, TEXTS)
    assert int(np.argmax(scores)) == 1
    assert scores[1] > 0.5 > max(scores[0], scores[2])


def test_one_score_per_text_between_0_and_1_in_input_order(reranker):
    scores = reranker.score(QUESTION, TEXTS)
    reversed_scores = reranker.score(QUESTION, TEXTS[::-1])
    assert scores.shape == (3,) and ((scores >= 0) & (scores <= 1)).all()
    np.testing.assert_allclose(scores, reversed_scores[::-1], atol=0.05)  # each score stays with its text


def test_batch_size_barely_changes_scores(reranker):
    texts = TEXTS * 4
    np.testing.assert_allclose(reranker.score(QUESTION, texts, 1), reranker.score(QUESTION, texts, 8), atol=0.05)


def test_empty_and_very_long_input(reranker):
    assert reranker.score(QUESTION, []).shape == (0,)
    long_text = "prompt injection defense " * 2000  # far beyond 512 tokens: must be cut, not crash
    assert reranker.score(QUESTION, [long_text]).shape == (1,)


def test_missing_model_gives_a_clear_message():
    with pytest.raises(SystemExit, match="export_reranker"):
        Reranker("no-such-model")


@pytest.mark.skipif(not (config.INDEX.exists() and config.DB.exists()), reason="index not built")
def test_search_with_reranking():
    from papertrail import cache
    from papertrail.retrieve import Retriever
    cache.enabled = False
    r = Retriever()
    plain = r.search(QUESTION, 40, mode="hybrid", rerank=False)
    reranked = r.search(QUESTION, 8, mode="hybrid", rerank=True)
    assert len(reranked) == 8
    scores = [p["score"] for p in reranked]
    assert scores == sorted(scores, reverse=True)
    # reranking only re-sorts the shortlist: every result comes from the 40 candidates
    ids = [p["id"] for p in plain]
    assert all(p["id"] == ids[p["search_rank"] - 1] for p in reranked)
    filtered = r.search(QUESTION, 5, category="cs.CR", rerank=True)
    assert filtered and all("cs.CR" in p["categories"] for p in filtered)
