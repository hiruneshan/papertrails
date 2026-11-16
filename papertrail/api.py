"""Step 9: a small HTTP API. Run with `papertrail serve`, then open http://localhost:8000/docs."""

from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import cache, config
from .generate import make_client, model_name
from .pipeline import ask as run_ask
from .retrieve import Retriever
from .timing import Timer, log_run

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup: this takes seconds, a search takes milliseconds.
    timer = Timer()
    state["retriever"] = Retriever(timer)
    state["client"] = make_client()
    log_run("api-startup", timer)
    yield
    state.clear()


app = FastAPI(title="PaperTrail", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    k: int = config.TOP_K
    category: str | None = None


def _timing(timer: Timer) -> dict:
    return {s: round(v * 1000, 1) for s, v in timer.stages.items()} | {"total": round(timer.total() * 1000, 1)}


@app.get("/health")
def health():
    return {"ok": True, "papers": state["retriever"].index.ntotal, "provider": config.LLM_PROVIDER,
            "model": model_name(), "cache": cache.available()}


@app.get("/search")
def search(q: str, k: int = config.TOP_K, category: str | None = None):
    """Retrieval only: the closest papers, no LLM call."""
    timer = Timer()
    papers = state["retriever"].search(q, k, category, timer)
    log_run("api-search", timer, question=q, k=k, category=category)
    return {"question": q, "papers": papers, "timing_ms": _timing(timer)}


@app.post("/ask")
def ask(req: AskRequest):
    """Retrieve papers, then have the LLM answer from them with citations."""
    timer = Timer()
    try:
        text, papers = run_ask(state["retriever"], state["client"], req.question, req.k, req.category, timer)
    except anthropic.AuthenticationError:
        raise HTTPException(500, "Invalid or missing ANTHROPIC_API_KEY")
    except anthropic.RateLimitError:
        raise HTTPException(429, "Rate limited by the Claude API, try again shortly")
    except anthropic.APIStatusError as e:
        raise HTTPException(502, f"Claude API error: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "Could not reach the Claude API")
    except Exception as e:  # OpenRouter errors not handled in generate.py
        raise HTTPException(502, f"{config.LLM_PROVIDER} error: {e}")
    log_run("api-ask", timer, question=req.question, k=req.k, category=req.category)
    return {"question": req.question, "answer": text, "papers": papers,
            "cached": timer.info.get("answer_cache_hit", False), "timing_ms": _timing(timer)}
