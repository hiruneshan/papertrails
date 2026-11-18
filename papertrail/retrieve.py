"""Step 7: turn a question into the most relevant papers."""

import duckdb
import faiss
import numpy as np

from . import cache, config
from .encoder import load_encoder
from .timing import Timer


class Retriever:
    """Loads the model, index and database once; then answers many searches."""

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

    def search(self, question: str, k: int = config.TOP_K, category: str | None = None,
               timer: Timer | None = None) -> list[dict]:
        timer = timer or Timer()

        # 1. Embed the question with the same model used for the papers (or reuse a cached vector).
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

        # 2. Find the nearest papers. Fetch extra when filtering by category,
        #    since some hits will be filtered out.
        with timer.stage("vector_search"):
            fetch = k * 10 if category else k
            scores, rows = self.index.search(qv, fetch)
            hits = {int(r): float(s) for r, s in zip(rows[0], scores[0]) if r != -1}
        if not hits:
            return []

        # 3. Turn row numbers into papers. A cursor per call keeps this safe
        #    when the API serves several requests at once.
        with timer.stage("db_lookup"):
            df = self.db.cursor().execute(
                "SELECT row_id, id, title, abstract, categories, update_date "
                "FROM papers WHERE row_id IN (SELECT unnest(?))",
                [list(hits)],
            ).df()
            df["score"] = df["row_id"].map(hits)
            df = df.sort_values("score", ascending=False)
            if category:
                keep = df["categories"].str.split().apply(lambda cats: any(c.startswith(category) for c in cats))
                df = df[keep]
            df["url"] = "https://arxiv.org/abs/" + df["id"]
            df["update_date"] = df["update_date"].astype(str)
            results = df.drop(columns="row_id").head(k).to_dict("records")
        return results
