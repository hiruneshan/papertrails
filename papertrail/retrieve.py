"""Step 7: turn a question into the most relevant papers."""

import duckdb
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from . import config


class Retriever:
    """Loads the model, index and database once; then answers many searches."""

    def __init__(self) -> None:
        for path in (config.INDEX, config.DB):
            if not path.exists():
                raise SystemExit(f"Missing {path}. Run `papertrail build-index` first.")
        torch.set_num_threads(config.CPU_THREADS)
        self.model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        self.index = faiss.read_index(str(config.INDEX))
        self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
        self.db = duckdb.connect(str(config.DB), read_only=True)

    def search(self, question: str, k: int = config.TOP_K, category: str | None = None) -> list[dict]:
        # 1. Embed the question with the same model used for the papers.
        qv = self.model.encode([config.QUERY_PREFIX + question], normalize_embeddings=True)
        qv = np.asarray(qv, dtype=np.float32)

        # 2. Find the nearest papers. Fetch extra when filtering by category,
        #    since some hits will be filtered out.
        fetch = k * 10 if category else k
        scores, rows = self.index.search(qv, fetch)
        hits = {int(r): float(s) for r, s in zip(rows[0], scores[0]) if r != -1}
        if not hits:
            return []

        # 3. Turn row numbers into papers. A cursor per call keeps this safe
        #    when the API serves several requests at once.
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
        return df.drop(columns="row_id").head(k).to_dict("records")
