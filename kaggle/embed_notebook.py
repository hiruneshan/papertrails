# PaperTrail - Kaggle notebook (steps 2-5): ingest, clean, embed.
# Paste each "# %% Cell" block into its own notebook cell, in order.
# Notebook settings: Accelerator = GPU T4 x2, Internet = On,
# input = "arXiv Dataset" by Cornell University.
# Outputs to download afterwards: /kaggle/working/emb/vectors.npy and papers.parquet

# %% Cell 1 - ingest: keep only computer science papers, write Parquet in batches
import json, os
import pandas as pd

SRC = "/kaggle/input/datasets/Cornell-University/arxiv/arxiv-metadata-oai-snapshot.json"
OUT = "/kaggle/working/parquet"
BATCH = 100_000
KEEP = ["id", "title", "abstract", "categories", "update_date"]

os.makedirs(OUT, exist_ok=True)

def is_cs(categories):
    # categories is a space-separated string, e.g. "cs.CL cs.LG stat.ML"
    return any(c.startswith("cs.") for c in categories.split())

def write_batch(rows, n):
    path = f"{OUT}/part-{n:04d}.parquet"
    pd.DataFrame(rows, columns=KEEP).to_parquet(path, index=False)

rows, part, total_read, total_kept = [], 0, 0, 0

with open(SRC, "r", encoding="utf-8") as f:
    for line in f:
        total_read += 1
        rec = json.loads(line)
        if not is_cs(rec["categories"]):
            continue
        rows.append([rec.get(k) for k in KEEP])
        total_kept += 1
        if len(rows) >= BATCH:
            write_batch(rows, part)
            part += 1
            rows = []
            print(f"part {part} written | read {total_read:,} | kept {total_kept:,}")

if rows:
    write_batch(rows, part)
    part += 1

print(f"DONE: read {total_read:,} papers, kept {total_kept:,} CS papers in {part} files")

# %% Cell 2 - clean, dedup, count tokens
import glob
import pandas as pd
from transformers import AutoTokenizer

IN = "/kaggle/working/parquet"
OUT = "/kaggle/working/clean.parquet"

df = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{IN}/*.parquet"))], ignore_index=True)
start = len(df)
print(f"loaded {start:,} papers")

for col in ["title", "abstract"]:
    df[col] = df[col].fillna("").str.replace(r"\s+", " ", regex=True).str.strip()

empty = df["abstract"].str.len() < 100
withdrawn = df["abstract"].str.contains("withdrawn", case=False) & (df["abstract"].str.len() < 300)
df = df[~empty & ~withdrawn]
print(f"dropped {int(empty.sum()):,} short/empty and {int((withdrawn & ~empty).sum()):,} withdrawn")

df["text"] = df["title"] + "\n\n" + df["abstract"]

before = len(df)
df = df.drop_duplicates(subset="text").reset_index(drop=True)
print(f"dropped {before - len(df):,} duplicates")

tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
texts = df["text"].tolist()
counts = []
for i in range(0, len(texts), 50_000):
    ids = tok(texts[i:i + 50_000], add_special_tokens=True, truncation=False)["input_ids"]
    counts.extend(len(x) for x in ids)
    print(f"tokenized {min(i + 50_000, len(texts)):,} / {len(texts):,}")
df["n_tokens"] = counts

df.to_parquet(OUT, index=False)
print(f"\nDONE: {start:,} -> {len(df):,} papers saved to {OUT}")
print(df["n_tokens"].describe().round(0))
print(f"papers over the model's 512-token limit: {(df['n_tokens'] > 512).sum():,} "
      f"({(df['n_tokens'] > 512).mean():.2%})")

# %% Cell 3 - embed on the GPU (set N = 10_000 for a quick test)
import os, glob, time
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer

N = None
SHARD = 100_000
OUT = "/kaggle/working/emb_test" if N else "/kaggle/working/emb"
os.makedirs(OUT, exist_ok=True)

df = pd.read_parquet("/kaggle/working/clean.parquet")
if N:
    df = df.head(N)
texts = df["text"].tolist()
print(f"{len(df):,} papers | GPU: {torch.cuda.get_device_name(0)}")

model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
model.half()
model.max_seq_length = 512

start = time.time()
for s, i in enumerate(range(0, len(texts), SHARD)):
    path = f"{OUT}/part-{s:04d}.npy"
    if os.path.exists(path):
        print(f"part {s} already done, skipping")
        continue
    t = time.time()
    vecs = model.encode(texts[i:i + SHARD], batch_size=256,
                        normalize_embeddings=True, convert_to_numpy=True)
    tmp = f"{OUT}/tmp-{s:04d}.npy"
    np.save(tmp, vecs.astype(np.float16))
    os.replace(tmp, path)
    secs = time.time() - t
    print(f"part {s}: {len(vecs):,} papers in {secs:.0f}s ({len(vecs) / secs:,.0f} papers/s)")

vectors = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{OUT}/part-*.npy"))])
assert len(vectors) == len(df), "vector count doesn't match paper count"
np.save(f"{OUT}/vectors.npy", vectors)
df.drop(columns=["text"]).to_parquet(f"{OUT}/papers.parquet", index=False)

print(f"\nDONE in {time.time() - start:.0f}s: vectors {vectors.shape}, "
      f"{vectors.nbytes / 1e6:.0f} MB, saved to {OUT}")

# %% Cell 4 - quick search check
q = "error correcting codes for wireless channels"
qv = model.encode(["Represent this sentence for searching relevant passages: " + q],
                  normalize_embeddings=True)[0]
scores = vectors.astype(np.float32) @ qv
for r in np.argsort(-scores)[:5]:
    print(f"{scores[r]:.3f}  {df.iloc[r]['id']}  {df.iloc[r]['title']}")
