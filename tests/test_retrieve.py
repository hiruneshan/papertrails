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


@pytest.fixture(scope="module")
def hybrid_retriever(retriever):
    if not retriever.has_keyword:
        pytest.skip("keyword search not built; run papertrail build-fts")
    return retriever


def test_keyword_search_finds_exact_terms(hybrid_retriever):
    """Keyword search must return papers that actually contain a rare acronym."""
    papers = hybrid_retriever.search("LoRA", k=10, mode="keyword")
    assert papers
    assert sum("lora" in (p["title"] + p["abstract"]).lower() for p in papers) >= 9


def test_hybrid_returns_both_kinds_and_valid_scores(hybrid_retriever):
    papers = hybrid_retriever.search("LoRA adapters for efficient fine-tuning of language models", k=8, mode="hybrid")
    assert len(papers) == 8
    assert {p["found_by"] for p in papers} <= {"both", "vector", "keyword"}
    scores = [p["score"] for p in papers]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_never_loses_a_paper_both_searches_agree_on(hybrid_retriever):
    q = "Raft consensus protocol for replicated logs"
    top_vector = {p["id"] for p in hybrid_retriever.search(q, k=3, mode="vector")}
    top_keyword = {p["id"] for p in hybrid_retriever.search(q, k=3, mode="keyword")}
    hybrid = {p["id"] for p in hybrid_retriever.search(q, k=8, mode="hybrid")}
    assert (top_vector & top_keyword) <= hybrid


def test_modes_fall_back_gracefully_on_empty_keywords(hybrid_retriever):
    # only stopwords: keyword search has nothing to match, hybrid must still answer from the vectors
    assert hybrid_retriever.search("the of and", k=5, mode="keyword") == []
    assert len(hybrid_retriever.search("the of and", k=5, mode="hybrid")) == 5
