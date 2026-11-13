"""Step 6: build the FAISS index and the DuckDB paper table from the Kaggle output."""

import time

import duckdb
import faiss
import numpy as np
import pandas as pd

from . import config
from .timing import Timer, log_run

ADD_BATCH = 100_000


def build() -> None:
    timer = Timer()
    for path in (config.VECTORS, config.PAPERS):
        if not path.exists():
            raise SystemExit(f"Missing {path}. Download it from the Kaggle notebook's output first.")

    # Vectors were saved as float16 to halve the download; FAISS needs float32.
    with timer.stage("load_vectors"):
        vectors = np.load(config.VECTORS)
        papers = pd.read_parquet(config.PAPERS)
    n, dim = vectors.shape
    print(f"loaded {n:,} vectors of {dim} numbers")

    if len(papers) != n:
        raise SystemExit(f"{len(papers):,} papers but {n:,} vectors - files don't match")

    # Row i of vectors.npy is paper i of papers.parquet. FAISS returns row numbers,
    # so row_id is how a search hit becomes a paper.
    with timer.stage("write_db"):
        papers.insert(0, "row_id", np.arange(n, dtype=np.int64))
        con = duckdb.connect(str(config.DB))
        con.register("papers_df", papers)
        con.execute("CREATE OR REPLACE TABLE papers AS SELECT * FROM papers_df")
        con.execute("CREATE INDEX IF NOT EXISTS papers_row ON papers(row_id)")
        con.close()
    del papers
    print(f"wrote paper details to {config.DB}")

    # HNSW: a graph where each vector links to its nearest neighbours.
    # Vectors are normalized, so inner product = cosine similarity.
    faiss.omp_set_num_threads(config.CPU_THREADS)
    index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION

    start = time.time()
    with timer.stage("build_hnsw"):
        for i in range(0, n, ADD_BATCH):
            index.add(vectors[i:i + ADD_BATCH].astype(np.float32))
            done = min(i + ADD_BATCH, n)
            rate = done / (time.time() - start)
            print(f"indexed {done:,} / {n:,}  ({rate:,.0f} vectors/s, ~{(n - done) / rate / 60:.0f} min left)")

    with timer.stage("save_index"):
        faiss.write_index(index, str(config.INDEX))
    print(f"wrote index to {config.INDEX} in {(time.time() - start) / 60:.1f} min")
    log_run("build-index", timer, vectors=n, index_mb=round(config.INDEX.stat().st_size / 1e6))
