"""Blind relevance check on real questions: does reranking put better papers in front of the LLM?

    ~/.venvs/papertrail/bin/python scripts/judge_rerank.py

For each question in bench/real_questions.txt, hybrid search picks its top 8 with and without the
reranker. The union is shuffled and an LLM grades every paper without being told which method chose it:
  2 = directly answers the question, 1 = related but doesn't really answer it, 0 = not relevant.
Uses one LLM call per question (free OpenRouter quota). The grades are an LLM's judgement, not a
human's, so treat small differences as noise. Results are saved to logs/judge_rerank.json.
"""

import json
import random
import re
import sys
import time

sys.path.insert(0, ".")
from papertrail import cache, config  # noqa: E402
from papertrail.generate import make_client, model_name  # noqa: E402
from papertrail.retrieve import Retriever  # noqa: E402

SYSTEM = """You grade search results for a research question. Grade each paper from its title and abstract only:
2 = directly addresses what the question asks
1 = on a related topic but does not really address what is asked
0 = not relevant
Pay attention to the question's intent (for example, defending against an attack is not the same as performing it).
Reply with only a JSON object mapping each paper number to its grade, like {"1": 2, "2": 0}."""


def grade(client, question, papers):
    listing = "\n\n".join(f"[{i}] {p['title']}\n{p['abstract']}" for i, p in enumerate(papers, 1))
    prompt = f"Question: {question}\n\nPapers:\n{listing}"
    for attempt in range(3):
        try:
            if config.LLM_PROVIDER == "openrouter":
                r = client.chat.completions.create(model=config.OPENROUTER_MODEL, max_tokens=4000, temperature=0,
                                                   messages=[{"role": "system", "content": SYSTEM},
                                                             {"role": "user", "content": prompt}])
                text = r.choices[0].message.content or ""
            else:
                r = client.messages.create(model=config.CLAUDE_MODEL, max_tokens=4000, system=SYSTEM,
                                           messages=[{"role": "user", "content": prompt}])
                text = "".join(b.text for b in r.content if b.type == "text")
            grades = json.loads(re.findall(r"\{[^{}]*\}", text)[-1])
            grades = {int(k): int(v) for k, v in grades.items()}
            if set(grades) == set(range(1, len(papers) + 1)) and set(grades.values()) <= {0, 1, 2}:
                return grades
            print("    incomplete grades, retrying", flush=True)
        except Exception as e:  # rate limits, bad JSON
            print(f"    attempt {attempt + 1} failed: {type(e).__name__}", flush=True)
        time.sleep(20)
    return None


cache.enabled = False
retriever = Retriever()
client = make_client()
questions = [q for q in open("bench/real_questions.txt").read().splitlines() if q.strip()]
rows = []
print(f"judge: {model_name()}\n{'question':62s} {'hybrid':>14s} {'+ reranker':>14s}   (grade total of 16 / papers graded 2)")
for n, q in enumerate(questions):
    a = retriever.search(q, 8, mode="hybrid", rerank=False)
    b = retriever.search(q, 8, mode="hybrid", rerank=True)
    pool = list({p["id"]: p for p in a + b}.values())
    random.Random(n).shuffle(pool)  # the judge can't tell which method a paper came from
    grades = grade(client, q, pool)
    if grades is None:
        print(f"{q[:60]:62s} skipped (no usable grades)", flush=True)
        continue
    by_id = {p["id"]: grades[i] for i, p in enumerate(pool, 1)}
    row = {"question": q}
    for name, result in (("hybrid", a), ("rerank", b)):
        g = [by_id[p["id"]] for p in result]
        row[name] = {"grades": g, "ids": [p["id"] for p in result], "total": sum(g), "twos": g.count(2), "zeros": g.count(0)}
    rows.append(row)
    print(f"{q[:60]:62s} {row['hybrid']['total']:>8d} / {row['hybrid']['twos']} {row['rerank']['total']:>8d} / {row['rerank']['twos']}", flush=True)

if rows:
    n = len(rows)
    print(f"\n{n} questions graded")
    for name, label in (("hybrid", "hybrid"), ("rerank", "hybrid + reranker")):
        print(f"  {label:18s} directly relevant {sum(r[name]['twos'] for r in rows) / (8 * n):6.1%}   "
              f"irrelevant {sum(r[name]['zeros'] for r in rows) / (8 * n):6.1%}   "
              f"mean grade {sum(r[name]['total'] for r in rows) / (8 * n):.2f} of 2")
    better = sum(r["rerank"]["total"] > r["hybrid"]["total"] for r in rows)
    worse = sum(r["rerank"]["total"] < r["hybrid"]["total"] for r in rows)
    print(f"  reranker better on {better}, worse on {worse}, same on {n - better - worse}")
    with open("logs/judge_rerank.json", "w") as f:
        json.dump({"judge": model_name(), "reranker": config.RERANK_MODEL, "rows": rows}, f, indent=1)
