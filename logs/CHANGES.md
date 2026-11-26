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

## v4-reranker

- **When:** 2026-11-26 14:38
- **Commit:** 76c4570+changes
- **What changed:** Cross-encoder reranker (ms-marco-MiniLM-L-6-v2, ONNX int8) re-sorts the best 40 hybrid-search candidates before the top 8 go to the LLM. RERANK=on to enable.
- **Settings:** provider=openrouter, llm_model=nvidia/nemotron-3-ultra-550b-a55b:free, embed_model=BAAI/bge-small-en-v1.5, embed_backend=onnx-int8, encoder_threads=8, top_k=8, hybrid=True, hybrid_candidates=50, rerank=False, rerank_model=minilm, rerank_candidates=40, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128, cpu_threads=8, cache=redis, prompt_version=1

**v4-reranker results** (2026-11-26) - off by default
- Speed (40 candidates): MiniLM int8 1.37 s per question (PyTorch 2.33 s); bge-base int8 6.9 s.
- `papertrail eval`, 200 questions each, right paper first: sentences 45.0% -> 96.0%, titles 93.5% -> 99.0%; search 0.1 s -> 1.5-1.7 s.
- Blind LLM grading on 12 real questions (top 8): directly relevant 69.8% -> 67.7%, irrelevant 10.4% -> 4.2%, mean grade 1.59 -> 1.64; better on 6, worse on 4. Within noise.
- 40 tests pass.

**User feedback with three-model review** (2026-11-26) - see docs/feedback.md
- Thumbs, 1-5 rating, reasons and comment on the chat page; stored in ~/rag-data/feedback.db.
- Negative ratings: analyst -> reviewer -> actor (three different free models); code limits actions, the only automatic change is deleting one cached answer.
- 69 tests pass. Real-model check (one run each): correct answer left alone, planted false claims caught and evicted.
