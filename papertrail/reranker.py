"""Cross-encoder reranker: re-sort candidate papers by reading the question and each paper together.

The search stages (vectors + keywords) are fast but never read the question next to a paper. A
cross-encoder does: it takes the pair and returns one relevance score. That is far more careful and far
slower, so it only runs on the few dozen candidates the search stages shortlisted.

Runs on ONNX Runtime with an int8 model made once by scripts/export_reranker.py (no PyTorch needed).
"""

import numpy as np

from . import config

MAX_TOKENS = 512  # question + paper together; longer abstracts are cut at the end


class Reranker:
    def __init__(self, name: str | None = None, int8: bool = True, threads: int | None = None) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        folder = config.RERANKER_DIR / (name or config.RERANK_MODEL)
        path = folder / "onnx" / ("model_qint8_avx512_vnni.onnx" if int8 else "model.onnx")
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/export_reranker.py once (see docs/setup.md), "
                             "or set RERANK=off.")
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        pad = "[PAD]" if self.tokenizer.token_to_id("[PAD]") is not None else "<pad>"
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id(pad), pad_token=pad)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or config.RERANK_THREADS
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def score(self, question: str, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """One relevance score between 0 and 1 for each text, in the order given."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        batch_size = batch_size or config.RERANK_BATCH
        # Batches are padded to their longest pair, so group texts of similar length to waste less work.
        by_length = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        logits = np.zeros(len(texts), dtype=np.float32)
        for start in range(0, len(texts), batch_size):
            ids = by_length[start:start + batch_size]
            encoded = self.tokenizer.encode_batch([(question, texts[i]) for i in ids])
            feed = {
                "input_ids": np.array([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in encoded], dtype=np.int64),
            }
            out = self.session.run(None, {k: v for k, v in feed.items() if k in self.inputs})[0]
            logits[ids] = out[:, 0]
        return 1.0 / (1.0 + np.exp(-logits))  # squash to 0..1; the order is unchanged
