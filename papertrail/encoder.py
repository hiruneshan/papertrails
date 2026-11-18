"""Question encoder: turns text into the same 384-number vectors as the paper embeddings.

EMBED_BACKEND in .env picks how the model runs:
  torch          sentence-transformers + PyTorch (the original; slow to import)
  onnx           ONNX Runtime, float32
  onnx-int8      ONNX Runtime, int8 weights (VNNI kernels)
  onnx-int8-avx2 ONNX Runtime, int8 weights (AVX2 kernels)
  openvino       OpenVINO, float32
  openvino-int8  OpenVINO, int8 (statically quantized)

All non-torch backends load files made once by scripts/export_encoder.py and need only
`tokenizers` plus ONNX Runtime or OpenVINO, so PyTorch is never imported at question time.
They reproduce the original model exactly: tokenize (lowercased, max 512 tokens),
run the transformer, keep the first ([CLS]) token's output, normalize to length 1.
"""

import numpy as np

from . import config

MODEL_FILES = {
    "onnx": "onnx/model.onnx",
    "onnx-int8": "onnx/model_qint8_avx512_vnni.onnx",
    "onnx-int8-avx2": "onnx/model_quint8_avx2.onnx",
    "openvino": "openvino/openvino_model.xml",
    "openvino-int8": "openvino/openvino_model_qint8_quantized.xml",
}
MAX_TOKENS = 512


class _ExportedEncoder:
    """Shared tokenize -> run -> CLS pooling -> normalize, for ONNX Runtime and OpenVINO."""

    def __init__(self, backend: str, threads: int) -> None:
        from tokenizers import Tokenizer

        folder = config.ENCODER_DIR
        path = folder / MODEL_FILES[backend]
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run scripts/export_encoder.py once (see docs/setup.md).")

        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=self.tokenizer.token_to_id("[PAD]"), pad_token="[PAD]")
        self._run = self._load_onnx(path, threads) if backend.startswith("onnx") else self._load_openvino(path, threads)

    @staticmethod
    def _load_onnx(path, threads):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        names = {i.name for i in session.get_inputs()}
        return lambda feed: session.run(None, {k: v for k, v in feed.items() if k in names})[0]

    @staticmethod
    def _load_openvino(path, threads):
        import openvino as ov

        compiled = ov.Core().compile_model(
            str(path), "CPU", {"INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY"})
        names = {i.any_name for i in compiled.inputs}
        output = compiled.outputs[0]
        return lambda feed: compiled({k: v for k, v in feed.items() if k in names})[output]

    def encode(self, texts: list[str]) -> np.ndarray:
        batch = self.tokenizer.encode_batch(texts)
        feed = {
            "input_ids": np.array([e.ids for e in batch], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in batch], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in batch], dtype=np.int64),
        }
        cls = np.asarray(self._run(feed))[:, 0, :].astype(np.float32)
        return cls / np.linalg.norm(cls, axis=1, keepdims=True)


class _TorchEncoder:
    def __init__(self, threads: int) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(threads)
        self.model = SentenceTransformer(config.EMBED_MODEL, device="cpu")

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def load_encoder(backend: str | None = None, threads: int | None = None):
    backend = (backend or config.EMBED_BACKEND).strip().lower()
    threads = threads or config.ENCODER_THREADS
    if backend == "torch":
        return _TorchEncoder(threads)
    if backend not in MODEL_FILES:
        raise SystemExit(f"Unknown EMBED_BACKEND {backend!r}; use torch or one of {', '.join(MODEL_FILES)}.")
    return _ExportedEncoder(backend, threads)
