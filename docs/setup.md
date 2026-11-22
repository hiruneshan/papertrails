# Setting up PaperTrail

Two parts: a **Kaggle notebook** that turns the papers into vectors (once, on a free GPU), and the **laptop** side that searches and answers.

## What you need

| Item | Why |
| --- | --- |
| Windows with **WSL2** (Ubuntu) | Everything on the laptop runs in Linux |
| **Python 3.11+** | Python 3.12 is used here |
| ~**6 GB free disk** inside WSL | Data files 1.5 GB + index 1.8 GB + database 0.7 GB + libraries |
| ~**4 GB free RAM** | The index is held in memory while searching |
| **Kaggle account**, phone-verified | Free GPU for embedding; verification unlocks GPU and internet in notebooks |
| **OpenRouter API key** (free), or an **Anthropic API key** (paid) | The LLM that writes answers. A Claude Pro subscription does not include API access. |

No NVIDIA GPU is needed on the laptop.

## Part 1 - Kaggle: make the vectors

1. **Create a notebook** at kaggle.com → Code → New Notebook.
2. **Add the data:** right panel → Add Input → search `arxiv` → **"arXiv Dataset" by Cornell University**.
3. **Settings** (right panel → Session options): Accelerator **GPU T4 x2**, Internet **On**.
4. **Paste the cells** from [`kaggle/embed_notebook.py`](../kaggle/embed_notebook.py), one `# %% Cell` block per notebook cell, in order:
   1. Ingest - keep computer science papers (~5 min)
   2. Clean - fix text, drop empty and duplicate papers, count tokens (~5 min)
   3. Embed - set `N = 10_000` for a quick test first, then `N = None` for all papers
   4. Search check - prints 5 papers for a sample question
5. **Download** `emb/vectors.npy` (765 MB) and `emb/papers.parquet` (728 MB).
   - From a saved version: Save Version → Save & Run All, then the version's **Output** tab.
   - From a live session: run `from IPython.display import FileLink; FileLink("emb/vectors.npy")` and click the link.

Files in `/kaggle/working/` are deleted when a live session stops. If they disappear, rerun the cells.

## Part 2 - Laptop: install

Open an Ubuntu (WSL) terminal:

```bash
cd "/mnt/c/Users/<you>/OneDrive/Desktop/Personal Dev/papertrail"

python3 -m venv ~/.venvs/papertrail
source ~/.venvs/papertrail/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build, much smaller
pip install -e .
```

Keep the environment and data in the Linux home folder (`~`), not under `/mnt/c`: reading files across the Windows boundary is much slower, and OneDrive would try to sync them.

## Part 3 - Laptop: data and keys

```bash
mkdir -p ~/rag-data
mv "/mnt/c/Users/<you>/Downloads/vectors.npy" "/mnt/c/Users/<you>/Downloads/papers.parquet" ~/rag-data/
cp .env.example .env
```

Open `.env` and fill in one key:

| Setting | Value |
| --- | --- |
| `LLM_PROVIDER` | `openrouter` (free) or `anthropic` (Claude) |
| `OPENROUTER_API_KEY` | from openrouter.ai/keys, starts with `sk-or-` |
| `OPENROUTER_MODEL` | any model ending in `:free` (default `nvidia/nemotron-3-ultra-550b-a55b:free`) |
| `ANTHROPIC_API_KEY` | from console.anthropic.com, starts with `sk-ant-` |
| `PAPERTRAIL_MODEL` | Claude model, e.g. `claude-opus-5-5` or the cheaper `claude-haiku-5-5` |
| `PAPERTRAIL_DATA` | data folder, default `~/rag-data` |
| `PAPERTRAIL_THREADS` | CPU threads, default 8 |
| `HYBRID` | `on` (default): keyword + vector search merged; `off`: vector only |
| `HYBRID_CANDIDATES` | papers each search contributes to the merge, default 50 |
| `EMBED_BACKEND` | question encoder: `onnx-int8` (default), `onnx`, `openvino`, `onnx-int8-avx2`, `openvino-int8`, `torch` |

`.env` is git-ignored. Never commit it or paste keys into chats.

## Part 4 - Export the question encoder (once)

Questions are embedded on the CPU with an int8 ONNX version of the model, which needs no PyTorch at question time. Export it once, in a separate environment (the export tools need older library versions than the app):

```bash
python3 -m venv ~/.venvs/papertrail-export
~/.venvs/papertrail-export/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
~/.venvs/papertrail-export/bin/pip install "sentence-transformers[onnx,openvino]" datasets "huggingface-hub>=0.34,<1.0"
~/.venvs/papertrail-export/bin/python scripts/export_encoder.py
```

This writes `~/rag-data/encoder/` (about 350 MB, all variants). To use the original PyTorch encoder instead, set `EMBED_BACKEND=torch` in `.env`.

## Part 5 - Build the index (once)

```bash
papertrail build-index
```

Takes about 7 minutes on an i7-13620H and creates `index_hnsw.faiss` and `papers.duckdb` in `~/rag-data`, then builds the keyword search for hybrid mode (about 4 more minutes; `papertrail build-fts` runs that part alone). If keyword search isn't built, PaperTrail says so and searches by meaning only.

## Part 5b - Redis cache (optional)

Redis caches question vectors and finished answers. Without it PaperTrail still works, it just never reuses anything.

```bash
sudo apt install -y redis-server        # needs sudo
```

No sudo? Build it in your home folder instead (needs `gcc` and `make`):

```bash
mkdir -p ~/redis-build && cd ~/redis-build
curl -fsSL -o redis.tar.gz https://download.redis.io/redis-stable.tar.gz && tar xzf redis.tar.gz
cd redis-stable && make -j8 MALLOC=libc          # the optional modules may fail to build; the server and CLI are enough
mkdir -p ~/redis/bin ~/redis/data && cp src/redis-server src/redis-cli ~/redis/bin/
```

Config at `~/redis/redis.conf`: `bind 127.0.0.1`, `daemonize yes`, `maxmemory 256mb`, `maxmemory-policy allkeys-lru`, `save ""`. Start it with `~/redis/bin/redis-server ~/redis/redis.conf` (again after each WSL restart) and check with `~/redis/bin/redis-cli ping`.

`papertrail cache-stats` shows what is cached; `papertrail cache-clear` empties it. Set `CACHE=off` in `.env` to run without it.

## Part 6 - Use it

Each new terminal needs:

```bash
cd "/mnt/c/Users/<you>/OneDrive/Desktop/Personal Dev/papertrail"
source ~/.venvs/papertrail/bin/activate
```

| Command | What it does |
| --- | --- |
| `papertrail search "quantum error correction"` | Closest papers, no LLM call |
| `papertrail ask "What are the main approaches to federated learning?"` | Cited answer + sources |
| `papertrail ask "..." --category cs.CV --k 5` | Only one category; fewer papers |
| `papertrail serve` | API at http://localhost:8000/docs; loads once, so each question skips the ~30 s startup |
| `papertrail bench` | Times 10 fixed questions × 3 runs (no LLM quota) |
| `papertrail eval --kind sentence` | Scores search quality: how often the right paper is found (vector, keyword, hybrid) |
| `papertrail report` | Median time per stage for each logged change |
| `papertrail log-change <label> "<what changed>"` | Records a change; later runs are tagged with it |

## Tests

```bash
~/.venvs/papertrail/bin/pip install pytest
~/.venvs/papertrail/bin/python -m pytest -q
```

Checks every encoder against the original model and runs real searches on the index (about 35 s).
`scripts/compare_encoders.py` reruns the full encoder comparison (accuracy and speed of every backend).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `429` / "model is busy or you've hit today's free limit" | Wait a minute, or set another `:free` model as `OPENROUTER_MODEL` |
| "Warning: You are sending unauthenticated requests to the HF Hub" | Harmless; it only checks for model updates |
| `Missing ... vectors.npy` | The data files are not in `PAPERTRAIL_DATA` (default `~/rag-data`) |
| Each command takes several seconds before searching | Loading the 1.8 GB index. Use `papertrail serve` to load once |
| `Missing .../encoder/...` | Run the one-time export in Part 4, or set `EMBED_BACKEND=torch` |
| Kaggle download fails for big files | Use the `FileLink` cell above, or download from a saved version's Output tab |
