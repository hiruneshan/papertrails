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

`.env` is git-ignored. Never commit it or paste keys into chats.

## Part 4 - Build the index (once)

```bash
papertrail build-index
```

Takes about 7 minutes on an i7-13620H and creates `index_hnsw.faiss` and `papers.duckdb` in `~/rag-data`.

## Part 5 - Use it

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
| `papertrail report` | Median time per stage for each logged change |
| `papertrail log-change <label> "<what changed>"` | Records a change; later runs are tagged with it |

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `429` / "model is busy or you've hit today's free limit" | Wait a minute, or set another `:free` model as `OPENROUTER_MODEL` |
| "Warning: You are sending unauthenticated requests to the HF Hub" | Harmless; it only checks for model updates |
| `Missing ... vectors.npy` | The data files are not in `PAPERTRAIL_DATA` (default `~/rag-data`) |
| Each command takes 30-60 s | Normal startup (libraries + 1.8 GB index). Use `papertrail serve` to load once |
| Kaggle download fails for big files | Use the `FileLink` cell above, or download from a saved version's Output tab |
