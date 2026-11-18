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
