"""One-time export of the question encoder to ONNX and OpenVINO, full precision and int8.

Run with the separate export environment (it needs PyTorch, optimum and datasets; the main app does not):
    ~/.venvs/papertrail-export/bin/python scripts/export_encoder.py

Writes to <PAPERTRAIL_DATA>/encoder/ (default ~/rag-data/encoder):
    tokenizer.json
    onnx/model.onnx                              ONNX, float32
    onnx/model_quint8_avx2.onnx                  ONNX, int8 weights (dynamic quantization, AVX2 kernels)
    onnx/model_qint8_avx512_vnni.onnx            ONNX, int8 weights (dynamic quantization, VNNI kernels)
    openvino/openvino_model.xml                  OpenVINO, float32
    openvino/openvino_model_qint8_quantized.xml  OpenVINO, int8 (static quantization, calibrated)
"""

import os
import shutil
import sys
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model,
    export_static_quantized_openvino_model,
)

MODEL = "BAAI/bge-small-en-v1.5"
OUT = Path(os.environ.get("PAPERTRAIL_DATA", "~/rag-data")).expanduser() / "encoder"
OUT.mkdir(parents=True, exist_ok=True)

print(f"exporting {MODEL} to {OUT}")

# ONNX float32: sentence-transformers converts the model on load, save_pretrained writes onnx/model.onnx
onnx_model = SentenceTransformer(MODEL, backend="onnx")
onnx_model.save_pretrained(str(OUT))
print("  onnx float32 done")

# ONNX int8: weights stored as 8-bit integers, activations quantized on the fly (no calibration data needed)
for cfg in ("avx2", "avx512_vnni"):
    export_dynamic_quantized_onnx_model(onnx_model, cfg, str(OUT))
    print(f"  onnx int8 ({cfg}) done")

# OpenVINO float32
ov_model = SentenceTransformer(MODEL, backend="openvino")
ov_model.save_pretrained(str(OUT))
print("  openvino float32 done")

# OpenVINO int8: static quantization, calibrated on sample sentences (default calibration set: GLUE sst2)
try:
    export_static_quantized_openvino_model(ov_model, quantization_config=None, model_name_or_path=str(OUT))
    print("  openvino int8 done")
except Exception as e:  # needs the `datasets` package and internet for calibration data
    print(f"  openvino int8 skipped: {e}", file=sys.stderr)

# The fast Rust tokenizer file is all the runtime needs to tokenize (save_pretrained writes it)
if not (OUT / "tokenizer.json").exists():
    onnx_model.tokenizer.save_pretrained(str(OUT))

print("\nfiles:")
for p in sorted(OUT.rglob("*")):
    if p.suffix in {".onnx", ".bin"} or p.name == "tokenizer.json":
        print(f"  {p.relative_to(OUT)}  {p.stat().st_size / 1e6:.1f} MB")
