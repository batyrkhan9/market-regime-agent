"""Score each headline in a raw news file with an LLM (Gemini Flash, free tier).

Headlines are sent in batches of BATCH_SIZE per request because the free tier caps
requests per day per model (20/day at the time of writing).

Input : data/raw/news_YYYY-MM-DD.json
Output: data/scored/scored_YYYY-MM-DD.json  (one {score, confidence, reason} per article)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

log = logging.getLogger("score_news")
logging.getLogger("google_genai").setLevel(logging.ERROR)  # silence per-call SDK chatter

# Pinned on purpose: changing the model changes the score distribution, so every
# output file records which model produced it and the prompt version it saw.
MODEL = "gemini-3.6-flash"
PROMPT_VERSION = "v2-batched"
RAW_DIR = Path("data/raw")
SCORED_DIR = Path("data/scored")
BATCH_SIZE = 25           # headlines per request; 181 headlines -> 8 requests
REQUEST_INTERVAL_S = 6.0  # free tier also caps requests per minute; pace ourselves

SYSTEM_PROMPT = """You are a financial news analyst scoring headlines for their implication
for the overall US stock market (S&P 500 / SPY) over the next few trading days.

You will receive a numbered list of news items. Return a JSON object with an "items" array
containing exactly one entry per input item, in the same order, each with:
- idx: the item's number from the input.
- score: float in [-1, 1]. -1 = strongly bearish for the broad US market, 0 = neutral or
  irrelevant to the broad market, +1 = strongly bullish.
- confidence: float in [0, 1]. How confident you are that the score is right AND that the
  news matters for the broad market. Single-stock news, celebrity news, or marketing
  pieces should get low confidence (<= 0.2) even if the score is non-zero.
- reason: one short sentence (<= 20 words) explaining the score.

Score each item independently. Judge only what the text says. Do not use knowledge of what
happened to markets afterwards."""

ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "idx": {"type": "integer"},
        "score": {"type": "number", "minimum": -1, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["idx", "score", "confidence", "reason"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": ITEM_SCHEMA}},
    "required": ["items"],
    "additionalProperties": False,
}


def format_batch(articles: list[dict]) -> str:
    """Turn a list of Alpaca articles into one numbered user message."""
    lines = []
    for i, a in enumerate(articles, start=1):
        summary = (a.get("summary") or "").strip() or "(none)"
        lines.append(f"{i}. Headline: {a['headline']}\n   Summary: {summary}")
    return "\n\n".join(lines)


def make_client() -> genai.Client:
    """Build a Gemini client that retries 429/5xx with exponential backoff."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing from environment (.env)")
    retry = types.HttpRetryOptions(attempts=8, initial_delay=5.0, max_delay=90.0,
                                   http_status_codes=[429, 500, 502, 503, 504])
    return genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=retry))


def score_batch(client: genai.Client, articles: list[dict]) -> list[dict]:
    """Ask the model for {score, confidence, reason} for each article, in input order."""
    response = client.models.generate_content(
        model=MODEL,
        contents=format_batch(articles),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_json_schema=OUTPUT_SCHEMA,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # fast classification
        ),
    )
    if not response.text:
        raise RuntimeError(f"Empty response for batch starting {articles[0]['id']}: {response}")
    items = {int(it["idx"]): it for it in json.loads(response.text)["items"]}
    if set(items) != set(range(1, len(articles) + 1)):
        raise RuntimeError(f"Model returned idx {sorted(items)} for {len(articles)} items")
    results = []
    for i in range(1, len(articles) + 1):
        it = items[i]
        results.append({
            "score": max(-1.0, min(1.0, float(it["score"]))),        # belt and braces:
            "confidence": max(0.0, min(1.0, float(it["confidence"]))),  # schema already bounds these
            "reason": it["reason"],
        })
    return results


# --- Anthropic alternative (swap make_client/score_batch; `uv add anthropic`) ----------
# import anthropic
# MODEL = "claude-opus-5"
#
# def make_client() -> anthropic.Anthropic:
#     return anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
#
# def score_batch(client: anthropic.Anthropic, articles: list[dict]) -> list[dict]:
#     response = client.messages.create(
#         model=MODEL,
#         max_tokens=4096,
#         system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
#         messages=[{"role": "user", "content": format_batch(articles)}],
#         output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
#     )
#     if response.stop_reason != "end_turn":
#         raise RuntimeError(f"Unexpected stop_reason={response.stop_reason}")
#     items = {int(it["idx"]): it for it in
#              json.loads(next(b.text for b in response.content if b.type == "text"))["items"]}
#     ...same idx check and clamping as above...
# ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="Date of the raw file to score (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="Re-score even if output exists")
    args = parser.parse_args()

    load_dotenv()
    raw_path = RAW_DIR / f"news_{args.date.isoformat()}.json"
    out_path = SCORED_DIR / f"scored_{args.date.isoformat()}.json"
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path} (run fetch_news.py first)")
    if out_path.exists() and not args.force:
        log.info("%s already exists; skipping (use --force to re-score)", out_path)
        return

    raw = json.loads(raw_path.read_text())
    articles = raw["news"]
    client = make_client()
    scored: list[dict] = []
    n_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE
    for b, start in enumerate(range(0, len(articles), BATCH_SIZE), start=1):
        batch = articles[start:start + BATCH_SIZE]
        t0 = time.monotonic()
        results = score_batch(client, batch)
        for article, result in zip(batch, results):
            scored.append({
                "id": article["id"],
                "created_at": article.get("created_at"),
                "headline": article["headline"],
                "source": article.get("source"),
                "url": article.get("url"),
                **result,
            })
        mean = sum(r["score"] for r in results) / len(results)
        log.info("batch %d/%d: %d headlines, mean score %+.2f", b, n_batches, len(batch), mean)
        if b < n_batches:
            time.sleep(max(0.0, REQUEST_INTERVAL_S - (time.monotonic() - t0)))

    record = {
        "date": raw["date"],
        "window_start": raw["window_start"],
        "window_end": raw["window_end"],
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_scored": len(scored),
        "scored": scored,
    }
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    log.info("Saved %d scored articles to %s (model=%s)", len(scored), out_path, MODEL)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
