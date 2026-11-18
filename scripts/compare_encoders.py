"""Check every question-encoder backend against the original PyTorch model.

    ~/.venvs/papertrail/bin/python scripts/compare_encoders.py [--queries 300]

For each backend it reports:
  vectors   cosine similarity of its question vectors to PyTorch's (1.0 = identical)
  search    how many of PyTorch's top-10 papers it also returns, and whether the top-1 paper matches
  correct   for title questions, how often the paper the title belongs to is ranked 1st / in the top 10
  speed     time to import + load in a fresh process, and median time to embed one question
"""

import argparse
import statistics
import subprocess
import sys
import time

import duckdb
import faiss
import numpy as np

sys.path.insert(0, ".")
from papertrail import config  # noqa: E402
from papertrail.encoder import MODEL_FILES, load_encoder  # noqa: E402

BACKENDS = ["torch", *MODEL_FILES]

parser = argparse.ArgumentParser()
parser.add_argument("--queries", type=int, default=300, help="paper titles used as test questions")
args = parser.parse_args()

# Test questions: the 10 benchmark questions + paper titles (fixed sample, so runs are comparable)
db = duckdb.connect(str(config.DB), read_only=True)
sample = db.execute(
    f"SELECT row_id, title FROM papers USING SAMPLE reservoir({args.queries} ROWS) REPEATABLE (42)").fetchall()
bench_qs = [q for q in open("bench/questions.txt").read().splitlines() if q.strip()]
questions = bench_qs + [t for _, t in sample]
truth = np.array([r for r, _ in sample])  # the right answer for each title question
texts = [config.QUERY_PREFIX + q for q in questions]

print(f"loading index ({config.INDEX.name}) ...", flush=True)
index = faiss.read_index(str(config.INDEX))
index.hnsw.efSearch = config.HNSW_EF_SEARCH


def embed_all(enc):
    """One question at a time, as in real use. Returns vectors and per-question milliseconds."""
    enc.encode(["warm up"])
    vecs, ms = [], []
    for t in texts:
        start = time.perf_counter()
        vecs.append(enc.encode([t]))
        ms.append((time.perf_counter() - start) * 1000)
    return np.vstack(vecs), ms


def startup_seconds(backend):
    """Import + load in a fresh Python process, the cost every cold start pays."""
    code = ("import time; t=time.perf_counter(); from papertrail.encoder import load_encoder; "
            f"e=load_encoder('{backend}'); e.encode(['x']); print(time.perf_counter()-t)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"}, cwd=".")
    return float(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 else float("nan")


results = {}
for b in BACKENDS:
    print(f"testing {b} ...", flush=True)
    enc = load_encoder(b)
    vecs, ms = embed_all(enc)
    _, top = index.search(vecs, 10)
    results[b] = {"vecs": vecs, "top": top, "ms": ms, "startup": startup_seconds(b)}
    del enc

ref = results["torch"]
n = len(texts)
print(f"\n{n} questions, threads={config.ENCODER_THREADS}\n")
print(f"{'backend':16s} {'vector sim':>11s} {'worst sim':>10s} {'top-10 same':>12s} {'top-1 same':>11s} "
      f"{'correct@1':>10s} {'correct@10':>11s} {'embed ms':>9s} {'startup s':>10s}")
for b, r in results.items():
    sim = (r["vecs"] * ref["vecs"]).sum(1)
    overlap = np.mean([len(set(a) & set(t)) / 10 for a, t in zip(r["top"], ref["top"])])
    top1 = np.mean(r["top"][:, 0] == ref["top"][:, 0])
    title_top = r["top"][len(bench_qs):]
    at1 = np.mean(title_top[:, 0] == truth)
    at10 = np.mean([t in row for t, row in zip(truth, title_top)])
    print(f"{b:16s} {sim.mean():11.5f} {sim.min():10.5f} {overlap:12.1%} {top1:11.1%} "
          f"{at1:10.1%} {at10:11.1%} {statistics.median(r['ms']):9.1f} {r['startup']:10.1f}")
