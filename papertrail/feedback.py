"""User feedback: what was answered, what people said about it, and what the review found.

Kept in a small SQLite file (Redis would lose it). Three tables:
  answers   a server-side snapshot of every answer shown, so feedback can only refer to real answers
  feedback  one row per visitor per answer (rating again replaces the earlier one)
  reviews   one automatic review per distinct answer, plus the human decision

Saving feedback changes nothing about search, answers or the cache. Only the review (agents.py) can
act, within the limits its code enforces.
"""

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta

from . import cache, config

REASONS = ("wrong", "off_topic", "bad_citation", "incomplete", "irrelevant_sources")
HUMAN_DECISIONS = ("confirmed", "rejected", "unclear")
MAX_COMMENT = 500

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    answer_id TEXT PRIMARY KEY, created TEXT NOT NULL, question TEXT NOT NULL, answer TEXT NOT NULL,
    papers TEXT NOT NULL, model TEXT, prompt_version INTEGER, cache_key TEXT, cached INTEGER, content_hash TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY, answer_id TEXT NOT NULL REFERENCES answers(answer_id), visitor TEXT NOT NULL,
    ip_hash TEXT, created TEXT NOT NULL, vote TEXT, rating INTEGER, reasons TEXT, comment TEXT, paper_votes TEXT,
    UNIQUE (answer_id, visitor));
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY, content_hash TEXT NOT NULL UNIQUE, answer_id TEXT NOT NULL, created TEXT NOT NULL,
    updated TEXT NOT NULL, status TEXT NOT NULL, report TEXT, action TEXT, human_decision TEXT, human_note TEXT);
"""


class FeedbackError(ValueError):
    """Feedback that must be refused; `status` is the HTTP code to answer with."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    config.FEEDBACK_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.FEEDBACK_DB, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        with con:  # commits on success, rolls back on error
            yield con
    finally:
        con.close()


def save_answer(question: str, answer: str, papers: list[dict], model: str, cache_key: str | None,
                cached: bool) -> str:
    """Record exactly what was shown. Returns the answer_id the browser must quote when rating."""
    answer_id = uuid.uuid4().hex
    slim = [{k: p.get(k) for k in ("id", "title", "abstract", "categories")} for p in papers]
    content = json.dumps([cache.normalize(question), answer, [p["id"] for p in slim]])
    with connect() as con:
        con.execute("INSERT INTO answers VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (answer_id, now(), question, answer, json.dumps(slim), model, config.PROMPT_VERSION,
                     cache_key, int(cached), hashlib.sha256(content.encode()).hexdigest()))
    return answer_id


def is_negative(vote, rating, paper_votes) -> bool:
    return vote == "down" or (rating is not None and rating <= 2) or "down" in (paper_votes or {}).values()


def save_feedback(answer_id: str, visitor: str, ip: str, vote: str | None = None, rating: int | None = None,
                  reasons: list[str] | None = None, comment: str = "", paper_votes: dict | None = None) -> dict:
    """Validate and store one visitor's current opinion of one answer, and queue a review if it is negative."""
    reasons, paper_votes, comment = reasons or [], paper_votes or {}, (comment or "").strip()
    if vote not in (None, "up", "down"):
        raise FeedbackError("vote must be up or down")
    if rating is not None and rating not in (1, 2, 3, 4, 5):
        raise FeedbackError("rating must be 1 to 5")
    if any(r not in REASONS for r in reasons):
        raise FeedbackError("unknown reason")
    if len(comment) > MAX_COMMENT:
        raise FeedbackError(f"comment is longer than {MAX_COMMENT} characters")
    if vote is None and rating is None and not paper_votes and not comment:
        raise FeedbackError("empty feedback")

    ip_hash = hashlib.sha256(ip.encode()).hexdigest()[:16]
    with connect() as con:
        answer = con.execute("SELECT * FROM answers WHERE answer_id = ?", (answer_id,)).fetchone()
        if answer is None:
            raise FeedbackError("unknown answer_id", 404)
        shown = {p["id"] for p in json.loads(answer["papers"])}
        if any(pid not in shown or v not in ("up", "down") for pid, v in paper_votes.items()):
            raise FeedbackError("paper votes must be up or down on papers shown with this answer")

        # Rate limit new ratings (changing an existing one doesn't count).
        existing = con.execute("SELECT 1 FROM feedback WHERE answer_id = ? AND visitor = ?",
                               (answer_id, visitor)).fetchone()
        if not existing:
            since = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
            count = con.execute("SELECT count(*) FROM feedback WHERE created >= ? AND (visitor = ? OR ip_hash = ?)",
                                (since, visitor, ip_hash)).fetchone()[0]
            if count >= config.FEEDBACK_PER_HOUR:
                raise FeedbackError("too much feedback from you in the last hour", 429)

        con.execute(
            """INSERT INTO feedback (answer_id, visitor, ip_hash, created, vote, rating, reasons, comment, paper_votes)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT (answer_id, visitor) DO UPDATE SET vote = excluded.vote, rating = excluded.rating,
                   reasons = excluded.reasons, comment = excluded.comment, paper_votes = excluded.paper_votes""",
            (answer_id, visitor, ip_hash, now(), vote, rating, json.dumps(sorted(set(reasons))), comment,
             json.dumps(paper_votes)))

        queued = False
        if is_negative(vote, rating, paper_votes):
            # One review per distinct answer: a hundred thumbs-down cost the same as one.
            queued = con.execute(
                "INSERT OR IGNORE INTO reviews (content_hash, answer_id, created, updated, status) VALUES (?,?,?,?,?)",
                (answer["content_hash"], answer_id, now(), now(), "pending")).rowcount == 1
    return {"saved": True, "review_queued": queued}


def list_reviews() -> list[dict]:
    """Reviews, ones waiting for a human first, with how many distinct visitors rated the answer down."""
    with connect() as con:
        rows = con.execute("""
            SELECT r.*, a.question,
                   (SELECT count(DISTINCT f.visitor) FROM feedback f JOIN answers a2 USING (answer_id)
                    WHERE a2.content_hash = r.content_hash AND f.vote = 'down') AS downvoters
            FROM reviews r JOIN answers a ON a.answer_id = r.answer_id
            ORDER BY (r.status = 'needs_human') DESC, downvoters DESC, r.id DESC""").fetchall()
    return [dict(r) for r in rows]


def get_review(review_id: int) -> dict | None:
    with connect() as con:
        review = con.execute("SELECT * FROM reviews WHERE id = ?", (review_id,)).fetchone()
        if review is None:
            return None
        answer = con.execute("SELECT * FROM answers WHERE answer_id = ?", (review["answer_id"],)).fetchone()
        ratings = con.execute("""SELECT f.vote, f.rating, f.reasons, f.comment, f.paper_votes FROM feedback f
                                 JOIN answers a USING (answer_id) WHERE a.content_hash = ?""",
                              (review["content_hash"],)).fetchall()
    return {"review": dict(review), "answer": dict(answer), "feedback": [dict(r) for r in ratings]}


def decide(review_id: int, decision: str, note: str = "") -> None:
    """The human's verdict; it is final."""
    if decision not in HUMAN_DECISIONS:
        raise FeedbackError(f"decision must be one of {', '.join(HUMAN_DECISIONS)}")
    with connect() as con:
        changed = con.execute("UPDATE reviews SET human_decision = ?, human_note = ?, status = 'closed', updated = ? "
                              "WHERE id = ?", (decision, note, now(), review_id)).rowcount
    if not changed:
        raise FeedbackError(f"no review {review_id}", 404)


def counts() -> dict:
    with connect() as con:
        one = lambda sql: con.execute(sql).fetchone()[0]
        return {
            "answers": one("SELECT count(*) FROM answers"),
            "up": one("SELECT count(*) FROM feedback WHERE vote = 'up'"),
            "down": one("SELECT count(*) FROM feedback WHERE vote = 'down'"),
            "reviews_pending": one("SELECT count(*) FROM reviews WHERE status = 'pending'"),
            "needs_human": one("SELECT count(*) FROM reviews WHERE status = 'needs_human'"),
        }
