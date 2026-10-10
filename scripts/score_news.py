"""Score each headline in a raw news file with an LLM.

Two providers are supported and selected with --provider (or SCORER_PROVIDER in .env):
  anthropic (default) - Claude Sonnet via the Anthropic API, key ANTHROPIC_API_KEY
  gemini              - Gemini Flash via google-genai (free tier), key GEMINI_API_KEY

Both providers see the identical SYSTEM_PROMPT, batch format and JSON schema, and write
the identical output rows, so scores from either can be compared. The output file records
which model produced it.

Headlines are sent in batches of BATCH_SIZE per request. Gemini additionally paces requests
because its free tier caps requests per minute and per day.

Input : data/raw/news_YYYY-MM-DD.json
Output: data/scored/scored_YYYY-MM-DD.json  (one {score, confidence, reason} per article)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

log = logging.getLogger("score_news")

# Pinned on purpose: changing the model changes the score distribution, so every
# output file records which model produced it and the prompt version it saw.
# "claude-sonnet-5-5" is the exact, complete model ID on the Anthropic API (there is no
# dated alias to pin to); "gemini-3.6-flash" likewise.
ANTHROPIC_MODEL = "claude-sonnet-5-5"
GEMINI_MODEL = "gemini-3.6-flash"
PROMPT_VERSION = "v2-batched"
RAW_DIR = Path("data/raw")
SCORED_DIR = Path("data/scored")
BATCH_SIZE = 25           # headlines per request; 181 headlines -> 8 requests
GEMINI_REQUEST_INTERVAL_S = 6.0  # free tier caps requests per minute; pace ourselves

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
# Anthropic structured outputs reject numeric minimum/maximum. The prompt already states
# the ranges and parse_items clamps, so the Anthropic request uses the schema without them.
ANTHROPIC_OUTPUT_SCHEMA = json.loads(json.dumps(OUTPUT_SCHEMA))
for _prop in ANTHROPIC_OUTPUT_SCHEMA["properties"]["items"]["items"]["properties"].values():
    _prop.pop("minimum", None)
    _prop.pop("maximum", None)


def format_batch(articles: list[dict]) -> str:
    """Turn a list of Alpaca articles into one numbered user message."""
    lines = []
    for i, a in enumerate(articles, start=1):
        summary = (a.get("summary") or "").strip() or "(none)"
        lines.append(f"{i}. Headline: {a['headline']}\n   Summary: {summary}")
    return "\n\n".join(lines)


def parse_items(text: str, n: int, batch_id: str) -> list[dict]:
    """Turn the model's JSON text into n {score, confidence, reason} dicts in input order."""
    if not text:
        raise RuntimeError(f"Empty response for batch starting {batch_id}")
    items = {int(it["idx"]): it for it in json.loads(text)["items"]}
    if set(items) != set(range(1, n + 1)):
        raise RuntimeError(f"Model returned idx {sorted(items)} for {n} items")
    results = []
    for i in range(1, n + 1):
        it = items[i]
        results.append({
            "score": max(-1.0, min(1.0, float(it["score"]))),           # belt and braces:
            "confidence": max(0.0, min(1.0, float(it["confidence"]))),  # schema already bounds these
            "reason": it["reason"],
        })
    return results


# --- Anthropic (Claude Sonnet) ------------------------------------------------------------

def make_anthropic_scorer() -> Callable[[list[dict]], list[dict]]:
    """Return a score_batch(articles) function backed by the Anthropic API."""
    import anthropic

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing from environment (.env)")
    # The SDK retries 429/5xx with backoff on its own; give it a few more attempts than
    # the default 2 so a brief rate limit does not abort the run.
    client = anthropic.Anthropic(max_retries=5)

    def score_batch(articles: list[dict]) -> list[dict]:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=16000,
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": format_batch(articles)}],
            # Lowest thinking setting on this model, mirroring Gemini's thinking_budget=0:
            # this is a fast classification task, not a reasoning one.
            thinking={"type": "between_tools"},
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": ANTHROPIC_OUTPUT_SCHEMA},
            },
        )
        if response.stop_reason != "end_turn":
            # "refusal" (safety classifier) or "max_tokens". We do not fall back to another
            # model, because the output must stay attributable to the pinned model.
            raise RuntimeError(
                f"Unexpected stop_reason={response.stop_reason} "
                f"details={response.stop_details} for batch starting {articles[0]['id']}"
            )
        text = next((b.text for b in response.content if b.type == "text"), "")
        return parse_items(text, len(articles), articles[0]["id"])

    return score_batch


# --- Gemini (Flash, free tier) ------------------------------------------------------------

def make_gemini_scorer() -> Callable[[list[dict]], list[dict]]:
    """Return a score_batch(articles) function backed by google-genai."""
    from google import genai
    from google.genai import types

    logging.getLogger("google_genai").setLevel(logging.ERROR)  # silence per-call SDK chatter
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY missing from environment (.env)")
    retry = types.HttpRetryOptions(attempts=8, initial_delay=5.0, max_delay=90.0,
                                   http_status_codes=[429, 500, 502, 503, 504])
    client = genai.Client(api_key=key, http_options=types.HttpOptions(retry_options=retry))

    def score_batch(articles: list[dict]) -> list[dict]:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=format_batch(articles),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_json_schema=OUTPUT_SCHEMA,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),  # fast classification
            ),
        )
        return parse_items(response.text or "", len(articles), articles[0]["id"])

    return score_batch


PROVIDERS = {
    "anthropic": (ANTHROPIC_MODEL, make_anthropic_scorer, 0.0),
    "gemini": (GEMINI_MODEL, make_gemini_scorer, GEMINI_REQUEST_INTERVAL_S),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="Date of the raw file to score (YYYY-MM-DD)")
    parser.add_argument("--provider", choices=sorted(PROVIDERS),
                        default=os.getenv("SCORER_PROVIDER", "anthropic"),
                        help="Which LLM to score with (default: $SCORER_PROVIDER or anthropic)")
    parser.add_argument("--force", action="store_true", help="Re-score even if output exists")
    args = parser.parse_args()

    load_dotenv()
    # argparse read the default before .env was loaded, so re-apply the env var if set there.
    if "--provider" not in sys.argv and os.getenv("SCORER_PROVIDER"):
        args.provider = os.environ["SCORER_PROVIDER"]
    if args.provider not in PROVIDERS:
        raise SystemExit(f"Unknown provider {args.provider!r}; choose from {sorted(PROVIDERS)}")
    model, make_scorer, request_interval_s = PROVIDERS[args.provider]

    raw_path = RAW_DIR / f"news_{args.date.isoformat()}.json"
    out_path = SCORED_DIR / f"scored_{args.date.isoformat()}.json"
    if not raw_path.exists():
        raise SystemExit(f"Raw file not found: {raw_path} (run fetch_news.py first)")
    if out_path.exists() and not args.force:
        log.info("%s already exists; skipping (use --force to re-score)", out_path)
        return

    raw = json.loads(raw_path.read_text())
    articles = raw["news"]
    score_batch = make_scorer()
    log.info("Scoring %d headlines with %s (%s)", len(articles), args.provider, model)
    scored: list[dict] = []
    n_batches = (len(articles) + BATCH_SIZE - 1) // BATCH_SIZE
    for b, start in enumerate(range(0, len(articles), BATCH_SIZE), start=1):
        batch = articles[start:start + BATCH_SIZE]
        t0 = time.monotonic()
        results = score_batch(batch)
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
            time.sleep(max(0.0, request_interval_s - (time.monotonic() - t0)))

    record = {
        "date": raw["date"],
        "window_start": raw["window_start"],
        "window_end": raw["window_end"],
        "model": model,
        "prompt_version": PROMPT_VERSION,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_scored": len(scored),
        "scored": scored,
    }
    SCORED_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    log.info("Saved %d scored articles to %s (model=%s)", len(scored), out_path, model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
