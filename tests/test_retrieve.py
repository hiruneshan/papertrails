"""End-to-end search over the real 1M-paper index (slow: loads the 1.8 GB index once)."""

import pytest

from papertrail import config

pytestmark = pytest.mark.skipif(not (config.INDEX.exists() and config.DB.exists()),
                                reason="index not built; run papertrail build-index")


@pytest.fixture(scope="module")
def retriever():
    from papertrail import cache
    from papertrail.retrieve import Retriever
    cache.enabled = False  # test the search itself, not cached vectors
    return Retriever()


def test_title_finds_its_own_paper(retriever):
    """An exact title should find its paper near the top. Not always first: for this title the original
    PyTorch model also ranks 'Formalizing and Benchmarking Prompt Injection Attacks and Defenses' first."""
    row = retriever.db.execute("SELECT id, title FROM papers WHERE id = '2505.18333'").fetchone()
    assert row, "test paper missing from the database"
    papers = retriever.search(row[1], k=10)
    assert row[0] in [p["id"] for p in papers[:3]]


def test_results_are_ranked_and_complete(retriever):
    papers = retriever.search("defending language models against prompt injection", k=8)
    assert len(papers) == 8
    scores = [p["score"] for p in papers]
    assert scores == sorted(scores, reverse=True)
    for p in papers:
        assert p["title"] and p["abstract"] and p["url"].startswith("https://arxiv.org/abs/")


def test_category_filter(retriever):
    papers = retriever.search("diffusion models for image generation", k=5, category="cs.CV")
    assert papers and all(any(c.startswith("cs.CV") for c in p["categories"].split()) for p in papers)
