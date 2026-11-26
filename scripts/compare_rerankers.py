"""Check the ONNX rerankers against the original PyTorch models and time them on real search results.

    ~/.venvs/papertrail/bin/python scripts/compare_rerankers.py [minilm bge]

For the 10 benchmark questions, hybrid search supplies 40 candidates each. Every variant scores the
same candidates. "top-8 same" is the share of PyTorch's best 8 papers that the variant also picks.
"""

import statistics
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from papertrail import cache, config  # noqa: E402
from papertrail.reranker import Reranker  # noqa: E402
from papertrail.retrieve import Retriever  # noqa: E402

HF_NAMES = {"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bge": "BAAI/bge-reranker-base"}
names = sys.argv[1:] or ["minilm", "bge"]

cache.enabled = False
config.RERANK = False  # gather candidates without reranking
retriever = Retriever()
questions = [q for q in open("bench/questions.txt").read().splitlines() if q.strip()]
cases = []
for q in questions:
    papers = retriever.search(q, config.RERANK_CANDIDATES, mode="hybrid")
    cases.append((q, [p["title"] + "\n\n" + p["abstract"] for p in papers]))
print(f"{len(cases)} questions x {len(cases[0][1])} candidates, {config.RERANK_THREADS} threads, batch {config.RERANK_BATCH}\n")


def run(score):
    score(cases[0][0], cases[0][1][:2])  # warm up
    all_scores, seconds = [], []
    for q, texts in cases:
        start = time.perf_counter()
        all_scores.append(np.asarray(score(q, texts)))
        seconds.append(time.perf_counter() - start)
    return all_scores, statistics.median(seconds)


print(f"{'model':8s} {'variant':14s} {'top-8 same':>10s} {'top-1 same':>10s} {'seconds / question':>19s}")
for name in names:
    from sentence_transformers import CrossEncoder
    torch_model = CrossEncoder(HF_NAMES[name], max_length=512, device="cpu")
    ref, ref_s = run(lambda q, t: torch_model.predict([(q, x) for x in t], batch_size=config.RERANK_BATCH))
    del torch_model
    print(f"{name:8s} {'pytorch':14s} {'-':>10s} {'-':>10s} {ref_s:19.2f}", flush=True)
    for label, int8 in (("onnx float32", False), ("onnx int8", True)):
        model = Reranker(name, int8=int8)
        got, secs = run(model.score)
        top8 = np.mean([len(set(np.argsort(-a)[:8]) & set(np.argsort(-b)[:8])) / 8 for a, b in zip(got, ref)])
        top1 = np.mean([np.argmax(a) == np.argmax(b) for a, b in zip(got, ref)])
        print(f"{name:8s} {label:14s} {top8:10.0%} {top1:10.0%} {secs:19.2f}", flush=True)
        del model
