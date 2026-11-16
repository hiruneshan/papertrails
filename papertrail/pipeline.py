"""The full question -> answer flow, with the answer cache. Used by the CLI and the API."""

from . import cache, config
from .generate import answer, model_name
from .timing import Timer


def ask(retriever, client, question: str, k: int, category: str | None, timer: Timer) -> tuple[str, list[dict]]:
    # Search always runs: it is cheap, and its result decides whether a cached answer is still valid.
    papers = retriever.search(question, k, category, timer)
    if not papers or not cache.enabled:
        return answer(question, papers, client, timer), papers

    key = cache.answer_key(question, [p["id"] for p in papers], k, category, config.LLM_PROVIDER, model_name())
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
