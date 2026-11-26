"""One-time export of cross-encoder rerankers to ONNX, full precision and int8.

Run with the export environment (the same one used for the question encoder):
    ~/.venvs/papertrail-export/bin/python scripts/export_reranker.py [minilm] [bge]

Writes to <PAPERTRAIL_DATA>/reranker/<name>/ (default ~/rag-data/reranker/):
    tokenizer.json
    onnx/model.onnx                     float32
    onnx/model_qint8_avx512_vnni.onnx   int8 weights (dynamic quantization, VNNI kernels)
"""

import os
import sys
from pathlib import Path

from sentence_transformers import CrossEncoder, export_dynamic_quantized_onnx_model

MODELS = {
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",  # small (~23M parameters)
    "bge": "BAAI/bge-reranker-base",                   # larger (~280M parameters)
}
ROOT = Path(os.environ.get("PAPERTRAIL_DATA", "~/rag-data")).expanduser() / "reranker"

for name in sys.argv[1:] or ["minilm"]:
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"exporting {MODELS[name]} to {out}", flush=True)
    model = CrossEncoder(MODELS[name], backend="onnx", model_kwargs={"export": True})
    model.save_pretrained(str(out))
    export_dynamic_quantized_onnx_model(model, "avx512_vnni", str(out))
    for p in sorted(out.rglob("*")):
        if p.suffix.startswith(".onnx") or p.name in ("tokenizer.json",) or p.name.endswith("onnx_data"):
            print(f"  {p.relative_to(out)}  {p.stat().st_size / 1e6:.1f} MB")
