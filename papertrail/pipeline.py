"""The full question -> answer flow, with the answer cache. Used by the CLI and the API."""

from . import cache, config
from .generate import answer, answer_stream, model_name
from .timing import Timer


def ask(retriever, client, question: str, k: int, category: str | None, timer: Timer) -> tuple[str, list[dict]]:
    # Search always runs: it is cheap, and its result decides whether a cached answer is still valid.
    papers = retriever.search(question, k, category, timer)
    if not papers or not cache.enabled:
        return answer(question, papers, client, timer), papers

    key = cache.answer_key(question, [p["id"] for p in papers], k, category, config.LLM_PROVIDER, model_name())
    timer.info["answer_key"] = key
    with timer.stage("answer_cache"):
        hit = cache.get_answer(key)
    if hit:
        timer.info.update(answer_cache_hit=True, answer_cached_at=hit["created"])
        return hit["answer"], papers
    timer.info["answer_cache_hit"] = False

    # If the same question is already being answered elsewhere, wait for that answer instead of a second LLM call.
    if not cache.acquire_lock(key):
        with timer.stage("answer_wait"):
            hit = cache.wait_for_answer(key)
        if hit:
            timer.info.update(answer_cache_hit=True, answer_cached_at=hit["created"], waited_for_other=True)
            return hit["answer"], papers

    try:
        text = answer(question, papers, client, timer)
        if "llm_error" not in timer.info:  # never cache errors, refusals or empty answers
            with timer.stage("answer_cache"):
                cache.put_answer(key, text, model_name())
    finally:
        cache.release_lock(key)
    return text, papers


def ask_stream(retriever, client, question: str, k: int, category: str | None, timer: Timer):
    """The same flow as ask(), as events for a streaming UI:
    ("papers", list) first, then ("delta", text) pieces, then ("cached", bool)."""
    papers = retriever.search(question, k, category, timer)
    yield "papers", papers
    if not papers or not cache.enabled:
        yield from (("delta", t) for t in answer_stream(question, papers, client, timer))
        yield "cached", False
        return

    key = cache.answer_key(question, [p["id"] for p in papers], k, category, config.LLM_PROVIDER, model_name())
    timer.info["answer_key"] = key
    with timer.stage("answer_cache"):
        hit = cache.get_answer(key)
    if not hit and not cache.acquire_lock(key):
        with timer.stage("answer_wait"):
            hit = cache.wait_for_answer(key)
        if hit:
            timer.info["waited_for_other"] = True
    if hit:
        timer.info.update(answer_cache_hit=True, answer_cached_at=hit["created"])
        yield "delta", hit["answer"]
        yield "cached", True
        return
    timer.info["answer_cache_hit"] = False

    parts = []
    try:
        for t in answer_stream(question, papers, client, timer):
            parts.append(t)
            yield "delta", t
        text = "".join(parts).strip()
        if text and "llm_error" not in timer.info:  # never cache errors, refusals or empty answers
            with timer.stage("answer_cache"):
                cache.put_answer(key, text, model_name())
    finally:
        cache.release_lock(key)
    yield "cached", False
