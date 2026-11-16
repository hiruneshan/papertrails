"""Redis cache with two layers.

L1  question -> question vector         key: embed model + question
    New papers never change a question's vector, so these survive corpus growth.

L2  answer                              key: question + the ids of the papers retrieved for it
                                             + LLM model + prompt version
    Search always runs (it is cheap). If it returns the same papers as last time, the
    cached answer is still valid. If new papers rank in, the ids differ, the key misses,
    and the LLM writes a fresh answer. No manual invalidation when papers are added.

If Redis is down or CACHE=off, every call quietly behaves as a miss.
"""

import hashlib
import json
import time
from datetime import datetime

import numpy as np

from . import config

PREFIX = "papertrail"
_client = None
_disabled_until = 0.0  # after a connection failure, skip Redis for a while instead of waiting on it every call
enabled = config.CACHE == "redis"  # bench turns this off to measure the raw pipeline


def _redis():
    global _client, _disabled_until
    if not enabled or time.monotonic() < _disabled_until:
        return None
    if _client is None:
        import redis
        _client = redis.Redis.from_url(config.REDIS_URL, socket_timeout=0.5, socket_connect_timeout=0.5)
    return _client


def _failed() -> None:
    global _disabled_until
    _disabled_until = time.monotonic() + 30


def available() -> bool:
    r = _redis()
    if r is None:
        return False
    try:
        return bool(r.ping())
    except Exception:
        _failed()
        return False


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


def _hash(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# ---------- L1: question vectors ----------

def qvec_key(question: str) -> str:
    model = config.EMBED_MODEL.split("/")[-1]
    return f"{PREFIX}:qvec:{model}:{_hash({'q': normalize(question)})}"


def get_qvec(question: str) -> np.ndarray | None:
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(qvec_key(question))
    except Exception:
        _failed()
        return None
    return None if raw is None else np.frombuffer(raw, dtype=np.float32).reshape(1, -1)


def put_qvec(question: str, vector: np.ndarray) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.set(qvec_key(question), np.asarray(vector, dtype=np.float32).tobytes(), ex=config.QVEC_TTL_DAYS * 86400)
    except Exception:
        _failed()


# ---------- L2: answers ----------

def answer_key(question: str, paper_ids: list[str], k: int, category: str | None,
               provider: str, llm_model: str) -> str:
    payload = {
        "q": normalize(question), "ids": paper_ids, "k": k, "category": category,
        "provider": provider, "llm": llm_model, "prompt_v": config.PROMPT_VERSION,
    }
    return f"{PREFIX}:answer:{_hash(payload)}"


def get_answer(key: str) -> dict | None:
    r = _redis()
    if r is None:
        return None
    try:
        raw = r.get(key)
    except Exception:
        _failed()
        return None
    return None if raw is None else json.loads(raw)


def put_answer(key: str, answer: str, llm_model: str) -> None:
    r = _redis()
    if r is None:
        return
    value = {"answer": answer, "llm": llm_model, "prompt_v": config.PROMPT_VERSION,
             "created": datetime.now().isoformat(timespec="seconds")}
    try:
        r.set(key, json.dumps(value), ex=config.ANSWER_TTL_DAYS * 86400)
    except Exception:
        _failed()


# ---------- lock: one LLM call per identical question at a time ----------

def acquire_lock(key: str, seconds: int = 120) -> bool:
    """True if this caller should compute the answer. Without Redis, always True."""
    r = _redis()
    if r is None:
        return True
    try:
        return bool(r.set(f"{key}:lock", "1", nx=True, ex=seconds))
    except Exception:
        _failed()
        return True


def release_lock(key: str) -> None:
    r = _redis()
    if r is None:
        return
    try:
        r.delete(f"{key}:lock")
    except Exception:
        _failed()


def wait_for_answer(key: str, timeout: float = 120.0) -> dict | None:
    """Another caller is computing this answer; wait for it to appear (or the lock to vanish)."""
    r = _redis()
    if r is None:
        return None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hit = get_answer(key)
        if hit:
            return hit
        try:
            if not r.exists(f"{key}:lock"):
                return None  # the other caller failed or gave up; compute it ourselves
        except Exception:
            _failed()
            return None
        time.sleep(0.25)
    return None


# ---------- admin ----------

def stats() -> dict:
    r = _redis()
    if r is None or not available():
        return {"available": False}
    counts = {"qvec": 0, "answer": 0}
    for key in r.scan_iter(f"{PREFIX}:*", count=1000):
        kind = key.decode().split(":")[1]
        if kind in counts and not key.endswith(b":lock"):
            counts[kind] += 1
    info = r.info()
    hits, misses = info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
    return {
        "available": True,
        "cached_question_vectors": counts["qvec"],
        "cached_answers": counts["answer"],
        "redis_memory": info.get("used_memory_human"),
        "redis_hit_rate_since_start": f"{hits / (hits + misses):.0%}" if hits + misses else "n/a",
    }


def clear() -> int:
    r = _redis()
    if r is None or not available():
        return 0
    keys = list(r.scan_iter(f"{PREFIX}:*", count=1000))
    return r.delete(*keys) if keys else 0
