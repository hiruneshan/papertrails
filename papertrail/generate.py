"""Step 8: have an LLM write an answer from the retrieved papers, with citations.

LLM_PROVIDER in .env picks the service:
  openrouter - free models through OpenRouter's OpenAI-compatible API
  anthropic  - Claude through the Anthropic SDK
"""

import anthropic

from . import config
from .timing import Timer

SYSTEM = """You answer questions about computer science research using only the arXiv papers given in the user's message.

- Base every claim on the provided papers and cite it inline as [arXiv:ID], using the paper's id.
- If the papers don't answer the question, say so plainly, then say what they do cover.
- Start with a short, direct answer, then give the supporting detail.
- Don't add facts from outside the provided papers."""

# These Claude models accept server-side refusal fallbacks; Haiku does not.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")


def make_client():
    """Create the client for whichever service LLM_PROVIDER names."""
    if config.LLM_PROVIDER == "openrouter":
        if not config.OPENROUTER_API_KEY:
            raise SystemExit("OPENROUTER_API_KEY is empty. Add your key to .env (https://openrouter.ai/keys).")
        from openai import OpenAI
        return OpenAI(base_url=config.OPENROUTER_URL, api_key=config.OPENROUTER_API_KEY)
    if config.LLM_PROVIDER == "anthropic":
        return anthropic.Anthropic()
    raise SystemExit(f"Unknown LLM_PROVIDER {config.LLM_PROVIDER!r}; use openrouter or anthropic.")


def model_name() -> str:
    return config.OPENROUTER_MODEL if config.LLM_PROVIDER == "openrouter" else config.CLAUDE_MODEL


def _format_papers(papers: list[dict]) -> str:
    parts = []
    for i, p in enumerate(papers, 1):
        parts.append(
            f'<paper index="{i}" id="{p["id"]}">\n'
            f'Title: {p["title"]}\n'
            f'Categories: {p["categories"]}\n'
            f'Last updated: {p["update_date"]}\n'
            f'Abstract: {p["abstract"]}\n'
            f"</paper>"
        )
    return "\n\n".join(parts)


def answer(question: str, papers: list[dict], client=None, timer: Timer | None = None) -> str:
    if not papers:
        return "No matching papers were found, so there is nothing to answer from."

    timer = timer or Timer()
    client = client or make_client()
    user_message = f"<papers>\n{_format_papers(papers)}\n</papers>\n\nQuestion: {question}"

    with timer.stage("llm"):
        if config.LLM_PROVIDER == "openrouter":
            text, usage = _answer_openrouter(client, user_message)
        else:
            text, usage = _answer_claude(client, user_message)
    timer.info.update(usage)
    return text


def answer_stream(question: str, papers: list[dict], client=None, timer: Timer | None = None):
    """Like answer(), but yields the text in pieces as the LLM writes it. Usage and errors land in timer.info."""
    if not papers:
        yield "No matching papers were found, so there is nothing to answer from."
        return

    timer = timer or Timer()
    client = client or make_client()
    user_message = f"<papers>\n{_format_papers(papers)}\n</papers>\n\nQuestion: {question}"

    with timer.stage("llm"):
        if config.LLM_PROVIDER == "openrouter":
            yield from _stream_openrouter(client, user_message, timer.info)
        else:
            yield from _stream_claude(client, user_message, timer.info)


def _stream_openrouter(client, user_message: str, info: dict):
    import openai
    try:
        stream = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            max_tokens=4000,
            stream=True,
            stream_options={"include_usage": True},
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
        got_text, finish = False, None
        for chunk in stream:
            if chunk.usage:
                info.update(input_tokens=chunk.usage.prompt_tokens, output_tokens=chunk.usage.completion_tokens)
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish = choice.finish_reason or finish
            if choice.delta and choice.delta.content:
                got_text = True
                yield choice.delta.content
    except openai.RateLimitError:
        info["llm_error"] = "rate_limit"
        yield (f"[{config.OPENROUTER_MODEL} is busy or you've hit today's free limit. "
               "Wait a minute and retry, or set another :free model as OPENROUTER_MODEL in .env.]")
        return
    except openai.AuthenticationError:
        info["llm_error"] = "auth"
        yield "[OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.]"
        return
    except (openai.NotFoundError, openai.PermissionDeniedError):
        info["llm_error"] = "unavailable"
        yield f"[{config.OPENROUTER_MODEL} isn't available. Pick another :free model for OPENROUTER_MODEL in .env.]"
        return
    if not got_text:
        info["llm_error"] = "empty"
        yield "The model returned an empty answer. Try again or pick another OPENROUTER_MODEL."
    elif finish == "length":
        yield "\n\n[answer cut off at the length limit]"


def _stream_claude(client: anthropic.Anthropic, user_message: str, info: dict):
    request = dict(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_message}],
    )
    if config.CLAUDE_MODEL in _FALLBACK_MODELS:
        manager = client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request)
    else:
        manager = client.messages.stream(**request)
    with manager as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    info.update(input_tokens=final.usage.input_tokens, output_tokens=final.usage.output_tokens)
    if final.stop_reason == "refusal":
        info["llm_error"] = "refusal"
        yield "\n\n[The model declined to answer this question.]"
    elif final.stop_reason == "max_tokens":
        yield "\n\n[answer cut off at the length limit]"


def _answer_openrouter(client, user_message: str) -> tuple[str, dict]:
    import openai
    try:
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            max_tokens=4000,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_message},
            ],
        )
    except openai.RateLimitError:
        return (f"[{config.OPENROUTER_MODEL} is busy or you've hit today's free limit. "
                "Wait a minute and retry, or set another :free model as OPENROUTER_MODEL in .env.]"), {"llm_error": "rate_limit"}
    except openai.AuthenticationError:
        return "[OpenRouter rejected the API key. Check OPENROUTER_API_KEY in .env.]", {"llm_error": "auth"}
    except (openai.NotFoundError, openai.PermissionDeniedError):
        return (f"[{config.OPENROUTER_MODEL} isn't available. "
                "Pick another :free model for OPENROUTER_MODEL in .env.]"), {"llm_error": "unavailable"}
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    if choice.finish_reason == "length":
        text += "\n\n[answer cut off at the length limit]"
    usage = {}
    if response.usage:
        usage = {"input_tokens": response.usage.prompt_tokens, "output_tokens": response.usage.completion_tokens}
    if not text:
        return "The model returned an empty answer. Try again or pick another OPENROUTER_MODEL.", usage | {"llm_error": "empty"}
    return text, usage


def _answer_claude(client: anthropic.Anthropic, user_message: str) -> tuple[str, dict]:
    request = dict(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_message}],
    )
    if config.CLAUDE_MODEL in _FALLBACK_MODELS:
        # If the model declines on a safety classifier, the API retries on a fallback model.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
        )
    else:
        response = client.messages.create(**request)

    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    if response.stop_reason == "refusal":
        return "The model declined to answer this question.", usage | {"llm_error": "refusal"}
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n[answer cut off at the length limit]"
    return text, usage
