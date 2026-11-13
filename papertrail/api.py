"""Step 9: a small HTTP API. Run with `papertrail serve`, then open http://localhost:8000/docs."""

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import config
from .generate import answer
from .retrieve import Retriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup: this takes seconds, a search takes milliseconds.
    state["retriever"] = Retriever()
    state["client"] = anthropic.Anthropic()
    yield
    state.clear()


app = FastAPI(title="PaperTrail", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int = config.TOP_K
    category: str | None = None


@app.get("/health")
def health():
    return {"ok": True, "papers": state["retriever"].index.ntotal, "model": config.CLAUDE_MODEL}


@app.get("/search")
def search(q: str, k: int = config.TOP_K, category: str | None = None):
    """Retrieval only: the closest papers, no LLM call."""
    return {"question": q, "papers": state["retriever"].search(q, k, category)}


@app.post("/ask")
def ask(req: AskRequest):
    """Retrieve papers, then have Claude answer from them with citations."""
    papers = state["retriever"].search(req.question, req.k, req.category)
    try:
        text = answer(req.question, papers, state["client"])
    except anthropic.AuthenticationError:
        raise HTTPException(500, "Invalid or missing ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the Claude API, try again shortly")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Claude API")
    return {"question": req.question, "answer": text, "papers": papers}
