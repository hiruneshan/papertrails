"""Every setting in one place. Override the environment ones in .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Files. vectors.npy and papers.parquet come from the Kaggle notebook;
# the index and database are built from them by `papertrail build-index`.
DATA_DIR = Path(os.environ.get("PAPERTRAIL_DATA", "~/rag-data")).expanduser()
VECTORS = DATA_DIR / "vectors.npy"
PAPERS = DATA_DIR / "papers.parquet"
INDEX = DATA_DIR / "index_hnsw.faiss"
DB = DATA_DIR / "papers.duckdb"

# Embedding model. Must be the same model the Kaggle notebook used.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on questions (not on papers).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# How the question encoder runs: torch, onnx, onnx-int8, onnx-int8-avx2, openvino, openvino-int8.
# Everything except torch uses files from scripts/export_encoder.py, stored in ENCODER_DIR.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "onnx-int8")
ENCODER_DIR = DATA_DIR / "encoder"

# HNSW index settings. M = links per node, efConstruction = build effort,
# efSearch = how widely each search explores (higher = better recall, slower).
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128

# How many papers to retrieve and hand to Claude.
TOP_K = 8

# Hybrid search: run keyword (BM25) and vector search, merge with rank fusion. HYBRID=off for vector only.
# HYBRID_CANDIDATES = how many papers each search contributes to the merge (more = better fusion, a little slower).
HYBRID = os.environ.get("HYBRID", "on").strip().lower() in ("on", "1", "true", "yes")
HYBRID_CANDIDATES = int(os.environ.get("HYBRID_CANDIDATES", "50"))

# Which service writes the answers: "openrouter" (free models) or "anthropic" (Claude).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").strip().lower()

# OpenRouter
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Claude
CLAUDE_MODEL = os.environ.get("PAPERTRAIL_MODEL", "claude-opus-5-5")

# CPU threads (i7-13620H has 16 threads; 8 is a good starting point).
CPU_THREADS = int(os.environ.get("PAPERTRAIL_THREADS", "8"))
# Threads for embedding one question (one short input often runs fastest on fewer threads).
ENCODER_THREADS = int(os.environ.get("ENCODER_THREADS", str(CPU_THREADS)))

# Cache (Redis). CACHE=off runs without it; if Redis is unreachable PaperTrail also runs without it.
CACHE = os.environ.get("CACHE", "redis").strip().lower()
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
ANSWER_TTL_DAYS = int(os.environ.get("ANSWER_TTL_DAYS", "30"))
QVEC_TTL_DAYS = int(os.environ.get("QVEC_TTL_DAYS", "90"))

# Bump when the LLM instructions in generate.py change, so old cached answers stop matching.
PROMPT_VERSION = 1
