"""Step 8: have Claude write an answer from the retrieved papers, with citations."""

import anthropic

from . import config

SYSTEM = """You answer questions about computer science research using only the arXiv papers given in the user's message.

- Base every claim on the provided papers and cite it inline as [arXiv:ID], using the paper's id.
- If the papers don't answer the question, say so plainly, then say what they do cover.
- Start with a short, direct answer, then give the supporting detail.
- Don't add facts from outside the provided papers."""

# These models accept server-side refusal fallbacks; Haiku does not.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-2")


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


def answer(question: str, papers: list[dict], client: anthropic.Anthropic | None = None) -> str:
    if not papers:
        return "No matching papers were found, so there is nothing to answer from."

    client = client or anthropic.Anthropic()
    request = dict(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": "medium"},
        messages=[{
            "role": "user",
            "content": f"<papers>\n{_format_papers(papers)}\n</papers>\n\nQuestion: {question}",
        }],
    )

    if config.CLAUDE_MODEL in _FALLBACK_MODELS:
        # If the model declines on a safety classifier, the API retries on a fallback model.
        response = client.beta.messages.create(
            betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
        )
    else:
        response = client.messages.create(**request)

    if response.stop_reason == "refusal":
        return "The model declined to answer this question."
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if response.stop_reason == "max_tokens":
        text += "\n\n[answer cut off at the length limit]"
    return text
