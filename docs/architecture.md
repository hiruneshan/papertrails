# PaperTrail architecture

PaperTrail answers questions about computer science research. It finds the arXiv papers closest in meaning to a question, then an LLM writes an answer from those papers' abstracts and cites each one.

- **Corpus:** 1,044,349 arXiv computer science papers (title + abstract)
- **Split:** the heavy one-time work (embedding) ran on a free Kaggle GPU; everything else runs on a laptop CPU

## System design

```mermaid
flowchart TB
  subgraph KAGGLE["Kaggle notebook - runs once, GPU"]
    direction LR
    RAW[("arXiv snapshot<br/>3,233,620 papers")] --> ING["1 · Ingest<br/>keep cs.* papers"]
    ING --> CLN["2 · Clean<br/>whitespace, drop empty<br/>and duplicates"]
    CLN --> EMB["3 · Embed<br/>bge-small, fp16, T4"]
  end

  EMB --> VEC[("vectors.npy<br/>1,044,349 × 384")]
  CLN --> PAP[("papers.parquet<br/>id, title, abstract,<br/>categories, date")]

  subgraph LAPTOP["Laptop - build once"]
    direction LR
    VEC --> IDX["4 · build-index<br/>FAISS HNSW"]
    PAP --> DBB["4 · build-index<br/>DuckDB table"]
  end

  IDX --> FAISS[("index_hnsw.faiss<br/>1.8 GB")]
  DBB --> DUCK[("papers.duckdb<br/>702 MB")]

  subgraph QUERY["Laptop - every question"]
    direction LR
    Q["question"] --> QE["5 · embed question<br/>same model"]
    QE --> SR["6 · nearest-neighbour<br/>search"]
    SR --> LK["7 · look up<br/>title + abstract"]
    LK --> LLM["8 · LLM writes answer<br/>with [arXiv:id] citations"]
  end

  FAISS -.-> SR
  DUCK -.-> LK
```

**How to read it:** vectors are only used to *find* papers. The LLM never sees vectors; it receives the original title and abstract text. Row *i* of `vectors.npy` is paper *i* of `papers.parquet`, and that row number is the link between FAISS (numbers) and DuckDB (text).

## What each step does

| Step | Where | What happens | Output |
| --- | --- | --- | --- |
| 1 · Ingest | Kaggle | Stream the 4–5 GB JSON snapshot line by line; keep papers with any `cs.*` category; write in batches of 100k | 11 Parquet files, 1,045,061 papers |
| 2 · Clean | Kaggle | Collapse whitespace, drop empty, short and withdrawn abstracts, drop exact duplicates, count tokens | `clean.parquet`, 1,044,349 papers |
| 3 · Embed | Kaggle | Turn each `title + abstract` into 384 numbers, in shards of 100k (finished shards are skipped on rerun) | `vectors.npy`, `papers.parquet` |
| 4 · Build index | Laptop | Link every vector to its nearest neighbours (HNSW); load paper text into DuckDB | `index_hnsw.faiss`, `papers.duckdb` |
| 5 · Embed question | Laptop | Same model, with bge's query prefix | 384 numbers |
| 6 · Search | Laptop | Walk the HNSW graph to the 8 closest papers | row numbers + similarity scores |
| 7 · Look up | Laptop | Fetch those rows' text from DuckDB; optional category filter | 8 papers |
| 8 · Answer | Internet | Send instructions + 8 abstracts + question to the LLM | cited answer |

**Why no chunking step:** the model reads at most 512 tokens. The median paper is 252 tokens and only 3,769 papers (0.36%) are longer, so each paper is one vector. Those few are cut at 512 tokens, losing the end of their abstract.

## Measured numbers

### Data

| Measure | Value |
| --- | --- |
| Papers in the arXiv snapshot | 3,233,620 |
| Computer science papers kept | 1,045,061 |
| After cleaning | 1,044,349 (712 removed) |
| Tokens per paper | median 252, mean 254, max 1,170 |
| Papers over the 512-token limit | 3,769 (0.36%) |

### Embedding on Kaggle

| Measure | Value |
| --- | --- |
| GPU | NVIDIA Tesla T4 (one of the two in "GPU T4 x2") |
| Model | `BAAI/bge-small-en-v1.5`, half precision (fp16), batch size 256 |
| Papers embedded | 1,044,349 |
| Output | 1,044,349 × 384 float16 = 765 MB |
| Speed (papers/s) and total time | **Not recorded yet** - from the `part N: ... papers/s` lines of the Kaggle run |

### Laptop (ASUS Vivobook 16, Intel i7-13620H, 16 GB RAM, 7.6 GB available to WSL)

| Measure | Value |
| --- | --- |
| Index build | 7.3 min (~2,400 vectors/s, 8 threads) |
| Index file | 1.8 GB |
| DuckDB file | 702 MB |
| Startup: import libraries | 12–16 s |
| Startup: load index from disk | 7–17 s |
| Startup: load embedding model | ~1 s |
| Per question: embed question | 190 ms (median) |
| Per question: FAISS search | 3 ms |
| Per question: DuckDB lookup | 22 ms |
| Per question: search total | 225 ms (median of 30) |
| Per question: LLM answer (free Nemotron 3 Ultra) | ~25 s (1 run) |

Timings come from `papertrail bench` (label `v0-baseline`). Every run is logged in `logs/timings.log`; changes are listed in `logs/CHANGES.md`.

**Where the time goes:** the LLM call and the one-time startup dominate. Search itself is under a quarter of a second, and most of that is embedding the question.

## Components: why we use them, and how they scale

### Kaggle notebook with a T4 GPU
**Why:** embedding a million papers is heavy matrix maths that a GPU does far faster than a laptop CPU. Kaggle gives a free GPU and already hosts the arXiv dataset, so nothing had to be downloaded first.
**At scale:** this is a batch job that runs once. New papers only need their own vectors, not a full rerun. For larger corpora the same script runs on any cloud GPU, and the 100k-paper shards can be split across several GPUs.

### Parquet files
**Why:** typed, compressed, column-based storage. Later steps read only the columns they need instead of re-parsing JSON text every time.
**At scale:** sharded Parquet is the standard input for parallel processing (several workers, one shard each), and DuckDB, Spark and pandas all read it directly.

### Embedding model: `BAAI/bge-small-en-v1.5`
**Why:** strong retrieval quality for its size (~33M parameters). Its 384 numbers per paper keep storage and search cheap.
**At scale:** storage and memory grow with the number of papers × dimensions, so small vectors keep 10M papers practical. Changing the model later means re-embedding every paper, and questions must always use the same model as the papers.

### FAISS with an HNSW index
**Why:** a library, not a server. HNSW finds nearest neighbours in about 3 ms here instead of comparing against all 1M vectors, and FAISS exposes every setting (`M`, `efSearch`) so speed and accuracy can be tuned and measured.
**At scale:** HNSW keeps every full vector in memory, about 1.8 GB per million papers here, so 10M papers would need about 18 GB of RAM. The next steps are compressed vectors (int8 or product quantization, which FAISS supports) to cut memory several times, and later a vector database such as Qdrant when filtering, updates or several servers matter.

### DuckDB
**Why:** an embedded SQL database stored in one file, with no server to run. It turns FAISS row numbers into titles and abstracts in about 22 ms.
**At scale:** lookups by row id stay fast at tens of millions of rows. It is built for one writer, so a system that adds papers continuously while serving would move this to Postgres or into a vector database that stores the text with each vector.

### LLM: OpenRouter (free) or Claude
**Why:** answer generation is one function with a setting (`LLM_PROVIDER`), so the model can change without touching retrieval. Free OpenRouter models cost nothing for testing; Claude is the paid option for better answers.
**At scale:** this is where the time and money go (~25 s per answer on the free model). Free models have daily limits and can be busy. A real deployment needs a paid model, per-question cost tracking, and caching of answers to repeated questions.

### FastAPI server
**Why:** loads the model and index once (~30 s), then answers every request with only the per-question cost. `/search` returns papers without an LLM call; `/ask` adds the answer.
**At scale:** each server process holds its own copy of the index in memory. Serving many users means memory-mapping the index or running search as a separate service shared by several web workers.

### Timing logs
**Why:** every run records how long each stage took and which change (label) it belongs to, so each optimization can be measured against the baseline instead of guessed.
**At scale:** the same records become monitoring: per-stage latency over time, and alerts when a stage slows down.

## Known limits

- Answers use **abstracts only**, not full papers. Questions about specific experiments or numbers inside a paper can't be answered.
- The category filter runs after the search (it fetches 80 candidates and keeps matches), so rare categories can return fewer than 8 papers.
- Free LLMs have daily request limits and are sometimes busy (HTTP 429).
