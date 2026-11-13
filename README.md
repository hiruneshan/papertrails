# PaperTrail

Ask questions about computer science research on arXiv. PaperTrail finds the most relevant
papers by meaning, then has Claude answer from them, citing each claim with its arXiv id.

Corpus: 1,044,349 arXiv computer science papers (title + abstract).

## How it works

| Step | Where | What happens |
| --- | --- | --- |
| 1-5 | Kaggle (GPU) | Load arXiv, keep CS papers, clean, embed each paper into 384 numbers - `kaggle/embed_notebook.py` |
| 6 | Laptop | Build a FAISS HNSW index + a DuckDB table of paper details - `papertrail build-index` |
| 7 | Laptop | Embed the question, find the nearest papers - `papertrail search` |
| 8 | Laptop | Claude answers from those papers with citations - `papertrail ask` |
| 9 | Laptop | HTTP API - `papertrail serve` |

## Setup (WSL)

```bash
python3 -m venv ~/.venvs/papertrail
source ~/.venvs/papertrail/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install -e .
cp .env.example .env    # then put your ANTHROPIC_API_KEY in .env
```

Data goes in `~/rag-data` (inside WSL, not under `/mnt/c` - that is much slower):

```bash
mkdir -p ~/rag-data
mv /mnt/c/Users/<you>/Downloads/vectors.npy /mnt/c/Users/<you>/Downloads/papers.parquet ~/rag-data/
```

## Use

```bash
papertrail build-index                                   # once, ~10-30 min
papertrail search "graph neural networks for molecules"  # retrieval only, free
papertrail ask "How do diffusion models compare to GANs for image generation?"
papertrail ask "What are common defenses against prompt injection?" --category cs.CR
papertrail serve                                         # then open http://localhost:8000/docs
```

## Settings

All in `papertrail/config.py`; the common ones can be set in `.env`:
`PAPERTRAIL_MODEL` (Claude model), `PAPERTRAIL_DATA` (data folder), `PAPERTRAIL_THREADS` (CPU threads).
