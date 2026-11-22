# PaperTrail change log

Each change gets a label; every run after it is tagged with that label in runs.jsonl and timings.log. Compare them with `papertrail report`.

## v0-baseline

- **When:** 2026-11-13 14:41
- **Commit:** fef2178+changes
- **What changed:** Starting point: bge-small fp32 on CPU (sentence-transformers), FAISS HNSW (M=32, efSearch=128) loaded fully into RAM, DuckDB lookup, free OpenRouter model for answers.
- **Settings:** provider=openrouter, llm_model=nvidia/nemotron-3-ultra-550b-a55b:free, embed_model=BAAI/bge-small-en-v1.5, top_k=8, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128, cpu_threads=8

## v1-onnx-int8

- **When:** 2026-11-17 21:04
- **Commit:** d5c422c+changes
- **What changed:** Question encoder runs on ONNX Runtime with an int8 model (VNNI kernels) via tokenizers + onnxruntime, without importing PyTorch. Exported once by scripts/export_encoder.py. Checked on 1,010 questions: same accuracy as PyTorch (right paper first 92.1% vs 92.0%, top 10 97.1% vs 97.0%). EMBED_BACKEND=torch restores the original.
- **Settings:** provider=openrouter, llm_model=nvidia/nemotron-3-ultra-550b-a55b:free, embed_model=BAAI/bge-small-en-v1.5, embed_backend=onnx-int8, encoder_threads=8, top_k=8, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128, cpu_threads=8, cache=redis, prompt_version=1

## v2-redis-cache

- **When:** 2026-11-17 21:34
- **Commit:** 2a509cb+changes
- **What changed:** Redis cache on: question vectors (L1) and answers keyed by question + retrieved paper ids + LLM model + prompt version (L2). Redis 8.10.2 built from source in ~/redis, 256 MB, least-recently-used eviction. Simple check: same question twice, second run answered from cache (LLM 21.2 s -> answer_cache 1 ms).
- **Settings:** provider=openrouter, llm_model=nvidia/nemotron-3-ultra-550b-a55b:free, embed_model=BAAI/bge-small-en-v1.5, embed_backend=onnx-int8, encoder_threads=8, top_k=8, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128, cpu_threads=8, cache=redis, prompt_version=1

## v3-hybrid-search

- **When:** 2026-11-20 16:20
- **Commit:** 34e8fa9+changes
- **What changed:** Hybrid search: BM25 keyword search (DuckDB full-text postings table, base-10 BM25, k1=1.2 b=0.75) plus vector search, merged with reciprocal rank fusion (k=60), 50 candidates from each. HYBRID=off restores vector-only.
- **Settings:** provider=openrouter, llm_model=nvidia/nemotron-3-ultra-550b-a55b:free, embed_model=BAAI/bge-small-en-v1.5, embed_backend=onnx-int8, encoder_threads=8, top_k=8, hybrid=True, hybrid_candidates=50, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128, cpu_threads=8, cache=redis, prompt_version=1

**v3-hybrid-search results** (2026-11-21)
- Quality, `papertrail eval` on 500 papers turned into questions, right paper in top 10: sentences 53.8% (vector) / 99.2% (keyword) / 95.4% (hybrid); titles 98.0% / 99.8% / 100.0%. The test is biased toward keyword search (each question contains the paper's own words), so it shows hybrid avoids misses, not how much it improves real questions. Hybrid ranks the right paper first less often than keyword-only (51.0% vs 97.0% on sentences).
- Speed, `papertrail bench`: search total 217 ms hybrid vs 54 ms vector-only (keyword step 171 ms median). Measured while the machine was slow (model load 10x slower than earlier); likely an upper bound.
- Keyword search built from DuckDB's full-text tables into a word-sorted `postings` table (75.3M rows, +0.8 GB, about 4 min to build). Same ranking as DuckDB's `match_bm25` on 7 queries, in 20-90 ms instead of 650-1200 ms.
- 34 tests pass (rank fusion, keyword search, hybrid).
