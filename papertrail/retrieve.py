"""Step 7: turn a question into the most relevant papers.

Three ways to search (mode):
  vector   nearest neighbours by meaning (FAISS)
  keyword  papers containing the question's words, ranked by BM25 (DuckDB postings table)
  hybrid   both, merged with reciprocal rank fusion

With reranking on (RERANK=on), whichever mode is used shortlists RERANK_CANDIDATES papers and a
cross-encoder re-sorts them before the best k are returned.
"""

import duckdb
import faiss
import numpy as np

from . import cache, config
from .encoder import load_encoder
from .fusion import rrf
from .reranker import Reranker
from .timing import Timer

# BM25 over the postings table built by `papertrail build-fts`. The stored weight w already holds each
# word's length-normalized, saturated term frequency; here we only multiply by the word's rarity (idf)
# and add up per paper. Words are lowercased, split and stemmed exactly as they were at build time,
# and common words like "the" and "of" are dropped.
KEYWORD_SQL = """
WITH tokens AS (
    SELECT DISTINCT stem(t, 'porter') AS t
    FROM (SELECT unnest(fts_main_papers.tokenize(?)) AS t)
    WHERE t <> '' AND t NOT IN (SELECT sw FROM fts_main_papers.stopwords)),
q AS (
    SELECT d.termid, log((st.num_docs - d.df + 0.5) / (d.df + 0.5) + 1) AS idf
    FROM fts_main_papers.dict d JOIN tokens ON d.term = tokens.t CROSS JOIN fts_main_papers.stats st)
SELECT p.docid AS row_id, sum(q.idf * p.w) AS score
FROM postings p JOIN q USING (termid)
GROUP BY p.docid ORDER BY score DESC LIMIT ?
"""


class Retriever:
    """Loads the encoder, index and database once; then answers many searches."""

    def __init__(self, timer: Timer | None = None) -> None:
        timer = timer or Timer()
        for path in (config.INDEX, config.DB):
            if not path.exists():
                raise SystemExit(f"Missing {path}. Run `papertrail build-index` first.")
        with timer.stage("load_model"):
            self.encoder = load_encoder()  # EMBED_BACKEND: onnx-int8 by default, PyTorch only if set to torch
        with timer.stage("load_index"):
            self.index = faiss.read_index(str(config.INDEX))
            self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
        with timer.stage("open_db"):
            self.db = duckdb.connect(str(config.DB), read_only=True)
            self.has_keyword = self._load_keyword_search()
        self.reranker = None
        if config.RERANK:
            with timer.stage("load_reranker"):
                self.reranker = Reranker()

    def _load_keyword_search(self) -> bool:
        try:
            self.db.execute("LOAD fts")
            self.db.execute("SELECT 1 FROM postings LIMIT 1")
            return True
        except duckdb.Error:
            if config.HYBRID:
                print("Keyword search is not built yet, so searching by meaning only. "
                      "Run `papertrail build-fts` to enable hybrid search.")
            return False

    def _mode(self, mode: str | None) -> str:
        mode = mode or ("hybrid" if config.HYBRID else "vector")
        if mode in ("keyword", "hybrid") and not self.has_keyword:
            return "vector"
        return mode

    def keyword_search(self, question: str, n: int) -> list[int]:
        """Row numbers of the n papers whose words best match the question (BM25), best first."""
        return [r[0] for r in self.db.cursor().execute(KEYWORD_SQL, [question, n]).fetchall()]

    def _embed(self, question: str, timer: Timer) -> np.ndarray:
        """The question as 384 numbers (or reuse a cached vector)."""
        qv = None
        if cache.enabled:
            with timer.stage("qvec_cache"):
                qv = cache.get_qvec(question)
            timer.info["qvec_cache_hit"] = qv is not None
        if qv is None:
            with timer.stage("embed_query"):
                qv = self.encoder.encode([config.QUERY_PREFIX + question])
            if cache.enabled:
                with timer.stage("qvec_cache"):
                    cache.put_qvec(question, qv)
        return qv

    def search(self, question: str, k: int = config.TOP_K, category: str | None = None,
               timer: Timer | None = None, mode: str | None = None, rerank: bool | None = None) -> list[dict]:
        timer = timer or Timer()
        mode = self._mode(mode)
        rerank = config.RERANK if rerank is None else rerank
        # Fetch extra when filtering by category, since some hits will be filtered out.
        want = k * 10 if category else k
        if rerank:  # the reranker needs a wider shortlist to choose from
            want = max(want, config.RERANK_CANDIDATES)
        # Hybrid reads a wider list from each search, so fusion can reward papers that rank well in both.
        depth = max(config.HYBRID_CANDIDATES, want) if mode == "hybrid" else want

        # 1. Rank candidate papers: [(row number, score, found_by), ...], best first.
        vector_rows: list[int] = []
        vector_scores: dict[int, float] = {}
        if mode in ("vector", "hybrid"):
            qv = self._embed(question, timer)
            with timer.stage("vector_search"):
                scores, rows = self.index.search(qv, depth)
                vector_rows = [int(r) for r in rows[0] if r != -1]
                vector_scores = {int(r): float(s) for r, s in zip(rows[0], scores[0]) if r != -1}
        keyword_rows: list[int] = []
        if mode in ("keyword", "hybrid"):
            with timer.stage("keyword_search"):
                keyword_rows = self.keyword_search(question, depth)

        if mode == "hybrid":
            with timer.stage("fusion"):
                ranked = [(row, score, "both" if row in vector_scores and row in set(keyword_rows)
                           else "vector" if row in vector_scores else "keyword")
                          for row, score in rrf(vector_rows, keyword_rows)[:want]]
        elif mode == "vector":
            ranked = [(row, vector_scores[row], "vector") for row in vector_rows[:want]]
        else:
            ranked = [(row, 0.0, "keyword") for row in keyword_rows[:want]]
        if not ranked:
            return []
        order = {row: i for i, (row, _, _) in enumerate(ranked)}
        info = {row: (score, found) for row, score, found in ranked}

        # 2. Turn row numbers into papers. A cursor per call keeps this safe
        #    when the API serves several requests at once.
        with timer.stage("db_lookup"):
            df = self.db.cursor().execute(
                "SELECT row_id, id, title, abstract, categories, update_date "
                "FROM papers WHERE row_id IN (SELECT unnest(?))",
                [list(order)],
            ).df()
            df["score"] = df["row_id"].map(lambda r: info[r][0])
            df["found_by"] = df["row_id"].map(lambda r: info[r][1])
            df = df.iloc[np.argsort(df["row_id"].map(order).to_numpy())]
            if category:
                keep = df["categories"].str.split().apply(lambda cats: any(c.startswith(category) for c in cats))
                df = df[keep]
            df["url"] = "https://arxiv.org/abs/" + df["id"]
            df["update_date"] = df["update_date"].astype(str)

        # 3. Optional: the cross-encoder reads the question with each shortlisted paper and re-sorts them.
        if rerank and len(df):
            if self.reranker is None:
                with timer.stage("load_reranker"):
                    self.reranker = Reranker()
            with timer.stage("rerank"):
                df = df.head(config.RERANK_CANDIDATES).copy()
                df["search_rank"] = np.arange(1, len(df) + 1)  # position before reranking
                df["score"] = self.reranker.score(question, (df["title"] + "\n\n" + df["abstract"]).tolist())
                df = df.sort_values("score", ascending=False, kind="stable")
        return df.drop(columns="row_id").head(k).to_dict("records")
