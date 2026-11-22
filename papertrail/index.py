"""Step 6: build the FAISS index and the DuckDB paper table from the Kaggle output."""

import time

import duckdb
import faiss
import numpy as np
import pandas as pd

from . import config
from .timing import Timer, log_run

ADD_BATCH = 100_000


def build_fts(timer: Timer | None = None) -> None:
    """Build the keyword (BM25) search over title + abstract inside papers.duckdb.

    Step 1, DuckDB's full-text extension tokenizes every paper into lowercase word stems
    ("training" and "trained" both become "train"), drops common words ("the", "of"), and
    records which paper contains which word.
    Step 2, we turn that into a `postings` table sorted by word. For each (word, paper) it
    stores that word's BM25 contribution (term frequency, normalized by the paper's length,
    saturated with k1=1.2, b=0.75). A search then reads only the rows of the question's words
    and adds them up, instead of scoring all 1M papers.
    """
    timer = timer or Timer()
    if not config.DB.exists():
        raise SystemExit(f"Missing {config.DB}. Run `papertrail build-index` first.")
    con = duckdb.connect(str(config.DB))
    con.execute("INSTALL fts")
    con.execute("LOAD fts")
    con.execute("PRAGMA memory_limit='4GB'")
    start = time.time()
    with timer.stage("build_fts"):
        con.execute(
            "PRAGMA create_fts_index('papers', 'row_id', 'title', 'abstract', "
            "stemmer='porter', stopwords='english', ignore='(\\.|[^a-z])+', strip_accents=1, lower=1, overwrite=1)"
        )
    print(f"  keyword index: {time.time() - start:.0f} s", flush=True)

    mid = time.time()
    with timer.stage("build_postings"):
        con.execute("DROP TABLE IF EXISTS postings")
        con.execute("""
            CREATE TABLE postings AS
            WITH tf AS (  -- how often each word occurs in each field (title / abstract) of each paper
                SELECT termid, docid, fieldid, count(*) AS tf FROM fts_main_papers.terms GROUP BY ALL),
            norm AS (     -- divide by how long that field is compared with the average (longer = weaker signal)
                SELECT tf.termid, tf.docid,
                       tf.tf / (0.25 + 0.75 * list_extract(d.field_lens, tf.fieldid + 1)
                                            / list_extract(st.avg_field_lens, tf.fieldid + 1)) AS ntf
                FROM tf JOIN fts_main_papers.docs d USING (docid) CROSS JOIN fts_main_papers.stats st),
            pseudo AS (SELECT termid, docid, sum(ntf) AS ptf FROM norm GROUP BY termid, docid)
            SELECT termid, docid, CAST(ptf * 2.2 / (ptf + 1.2) AS FLOAT) AS w   -- saturation: k1 = 1.2
            FROM pseudo ORDER BY termid, docid                                  -- sorted by word: lookups skip the rest
        """)
    rows = con.execute("SELECT count(*) FROM postings").fetchone()[0]
    con.close()
    print(f"  postings table: {rows:,} rows in {time.time() - mid:.0f} s")
    print(f"built keyword search in {time.time() - start:.0f} s; database is now {config.DB.stat().st_size / 1e6:.0f} MB")
    log_run("build-fts", timer)


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
    build_fts()  # the table was recreated above, so the keyword index must be rebuilt
