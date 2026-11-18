"""Question encoder tests: every backend must reproduce the original PyTorch model's vectors.

Needs the exported encoder files and the data in PAPERTRAIL_DATA; tests skip if they are missing.
Run:  ~/.venvs/papertrail/bin/python -m pytest -q
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from papertrail import config
from papertrail.encoder import MODEL_FILES, load_encoder

pytestmark = pytest.mark.skipif(not (config.ENCODER_DIR / "tokenizer.json").exists(),
                                reason="encoder not exported; run scripts/export_encoder.py")

QUESTIONS = [config.QUERY_PREFIX + q for q in [
    "How do people defend large language models against prompt injection attacks?",
    "graph neural networks for molecules",
    "LoRA",                                             # very short
    "What is the time complexity of Dijkstra's algorithm with a Fibonacci heap?",
    "Ünïcödé, punctuation!? and CAPITALS in a question",  # normalization
]]
LONG_TEXT = config.QUERY_PREFIX + " ".join(["transformer attention memory efficiency"] * 300)  # > 512 tokens

# Minimum cosine similarity to the PyTorch vectors (1.0 = identical). float32 exports must be exact;
# int8 exports may differ slightly (measured worst cases: 0.986, 0.978 and 0.953 over 1,010 questions).
MIN_SIMILARITY = {
    "onnx": 0.9999, "openvino": 0.9999,
    "onnx-int8": 0.97, "onnx-int8-avx2": 0.97, "openvino-int8": 0.94,
}
# int8 error grows with input length: on a 512-token text ONNX int8 measured 0.952 (VNNI) and 0.945 (AVX2).
# Questions are short, so this does not affect search, but it is why the int8 encoder is for questions
# only; papers are embedded at full precision on Kaggle.
MIN_SIMILARITY_LONG = {
    "onnx": 0.9999, "openvino": 0.9999,
    "onnx-int8": 0.93, "onnx-int8-avx2": 0.93, "openvino-int8": 0.97,
}
FLOAT32 = {"onnx", "openvino"}


@pytest.fixture(scope="module")
def reference():
    torch_enc = load_encoder("torch")
    return torch_enc.encode(QUESTIONS), torch_enc.encode([LONG_TEXT])


@pytest.fixture(scope="module", params=list(MODEL_FILES))
def encoder(request):
    return request.param, load_encoder(request.param)


def test_shape_dtype_and_unit_length(encoder):
    _, enc = encoder
    v = enc.encode(QUESTIONS)
    assert v.shape == (len(QUESTIONS), 384)
    assert v.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


def test_matches_pytorch(encoder, reference):
    name, enc = encoder
    ref, _ = reference
    sims = [(enc.encode([q]) * r).sum() for q, r in zip(QUESTIONS, ref)]  # one at a time, like real use
    assert min(sims) >= MIN_SIMILARITY[name], f"{name}: similarities {np.round(sims, 5)}"


def test_batch_equals_single(encoder):
    """Padding a batch must not change any vector (the attention mask has to be respected).

    float32 must match exactly. Dynamic int8 sets its scale from the whole batch, so batched vectors
    shift slightly (measured min similarity 0.995); the app embeds one question at a time.
    """
    name, enc = encoder
    batch = enc.encode(QUESTIONS)
    single = np.vstack([enc.encode([q]) for q in QUESTIONS])
    if name in FLOAT32:
        np.testing.assert_allclose(batch, single, atol=1e-4)
    else:
        assert (batch * single).sum(1).min() >= MIN_SIMILARITY[name]


def test_long_text_truncated_like_pytorch(encoder, reference):
    name, enc = encoder
    _, ref_long = reference
    v = enc.encode([LONG_TEXT])
    assert len(enc.tokenizer.encode(LONG_TEXT).ids) == 512
    assert (v * ref_long).sum() >= MIN_SIMILARITY_LONG[name]


def test_unknown_backend_is_rejected():
    with pytest.raises(SystemExit):
        load_encoder("no-such-backend")


def test_no_pytorch_at_question_time():
    """The app's search and answer code must not import PyTorch or sentence-transformers."""
    code = ("import sys; import papertrail.cli, papertrail.retrieve, papertrail.pipeline, papertrail.api; "
            "from papertrail.encoder import load_encoder; load_encoder('onnx-int8').encode(['x']); "
            "bad = [m for m in ('torch', 'sentence_transformers', 'transformers') if m in sys.modules]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "HF_HUB_OFFLINE": "1"})
    assert out.returncode == 0, f"imported: {out.stdout.strip()} {out.stderr[-500:]}"
