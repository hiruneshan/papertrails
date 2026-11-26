"""Step 9: a small HTTP API. Run with `papertrail serve`, then open http://localhost:8000 (chat UI) or /docs."""

import json
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from . import agents, cache, config, feedback
from .generate import make_client, model_name
from .pipeline import ask as run_ask, ask_stream as run_ask_stream
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


def _snapshot(question: str, text: str, papers: list[dict], timer: Timer) -> str | None:
    """Save what was shown so it can be rated. No id (so no rating) for errors or empty results;
    a storage problem must never break answering."""
    if not text or not papers or "llm_error" in timer.info:
        return None
    try:
        return feedback.save_answer(question, text, papers, model_name(), timer.info.get("answer_key"),
                                    bool(timer.info.get("answer_cache_hit")))
    except Exception as e:
        print(f"could not save the answer for feedback: {e}")
        return None


class FeedbackRequest(BaseModel):
    answer_id: str
    vote: str | None = None
    rating: int | None = None
    reasons: list[str] = []
    comment: str = ""
    papers: dict[str, str] = {}


@app.post("/feedback")
def post_feedback(req: FeedbackRequest, request: Request, response: Response, background: BackgroundTasks):
    """Store a rating. Negative ratings queue a three-model review that runs after this reply."""
    visitor = request.cookies.get("pt_visitor", "")
    if not re.fullmatch(r"[0-9a-f]{32}", visitor):
        visitor = uuid.uuid4().hex
        response.set_cookie("pt_visitor", visitor, max_age=365 * 86400, httponly=True, samesite="strict")
    try:
        result = feedback.save_feedback(req.answer_id, visitor, request.client.host if request.client else "",
                                        req.vote, req.rating, req.reasons, req.comment, req.papers)
    except feedback.FeedbackError as e:
        raise HTTPException(e.status, str(e))
    if result["review_queued"] and config.FEEDBACK_REVIEW:
        background.add_task(agents.process_pending, 1)
    return result


@app.get("/", include_in_schema=False)
def index():
    """The chat UI: one static page that calls /ask/stream."""
    return FileResponse(Path(__file__).parent / "static" / "index.html")


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
            "answer_id": _snapshot(req.question, text, papers, timer),
            "cached": timer.info.get("answer_cache_hit", False), "timing_ms": _timing(timer)}


def _error_detail(e: Exception) -> str:
    if isinstance(e, anthropic.AuthenticationError):
        return "Invalid or missing ANTHROPIC_API_KEY"
    if isinstance(e, anthropic.RateLimitError):
        return "Rate limited by the Claude API, try again shortly"
    if isinstance(e, anthropic.APIStatusError):
        return f"Claude API error: {e.message}"
    if isinstance(e, anthropic.APIConnectionError):
        return "Could not reach the Claude API"
    return f"{config.LLM_PROVIDER} error: {e}"


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Like /ask, but streams the reply as newline-delimited JSON events, so the UI can show it as it is written:
    {"type": "papers", "papers": [...]}, then {"type": "delta", "text": "..."} pieces,
    then {"type": "done", "cached": bool, "timing_ms": {...}} or {"type": "error", "detail": "..."}."""
    def line(event: dict) -> str:
        return json.dumps(jsonable_encoder(event), default=float) + "\n"

    def events():
        timer = Timer()
        cached = False
        shown, parts = [], []
        try:
            for kind, value in run_ask_stream(state["retriever"], state["client"], req.question, req.k,
                                              req.category, timer):
                if kind == "papers":
                    shown = value
                    yield line({"type": "papers", "papers": value})
                elif kind == "delta":
                    parts.append(value)
                    yield line({"type": "delta", "text": value})
                else:
                    cached = value
        except Exception as e:  # headers are already sent, so errors travel as an event
            yield line({"type": "error", "detail": _error_detail(e)})
            return
        log_run("api-ask-stream", timer, question=req.question, k=req.k, category=req.category)
        yield line({"type": "done", "cached": cached, "timing_ms": _timing(timer),
                    "answer_id": _snapshot(req.question, "".join(parts).strip(), shown, timer)})

    # X-Accel-Buffering stops proxies such as nginx from holding the stream back.
    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
