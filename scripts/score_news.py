"""Score each headline in a raw news file with Claude.

Input : data/raw/news_YYYY-MM-DD.json
Output: data/scored/scored_YYYY-MM-DD.json  (one {score, confidence, reason} per article)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path

import anthropic
from dotenv import load_dotenv

log = logging.getLogger("score_news")

# Pinned on purpose: changing the model changes the score distribution, so every
# output file records which model produced it and the prompt version it saw.
MODEL = "claude-opus-5"
PROMPT_VERSION = "v1"
RAW_DIR = Path("data/raw")
SCORED_DIR = Path("data/scored")

SYSTEM_PROMPT = """You are a financial news analyst scoring headlines for their implication
for the overall US stock market (S&P 500 / SPY) over the next few trading days.

Return a JSON object with:
- score: float in [-1, 1]. -1 = strongly bearish for the broad US market, 0 = neutral or
  irrelevant to the broad market, +1 = strongly bullish.
- confidence: float in [0, 1]. How confident you are that the score is right AND that the
  news matters for the broad market. Single-stock news, celebrity news, or marketing
  pieces should get low confidence (<= 0.2) even if the score is non-zero.
- reason: one short sentence (<= 20 words) explaining the score.

Judge only what the text says. Do not use knowledge of what happened to markets afterwards."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "minimum": -1, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["score", "confidence", "reason"],
    "additionalProperties": False,
}


def format_article(article: dict) -> str:
    """Turn one Alpaca article into the user message the model scores."""
    summary = (article.get("summary") or "").strip()
    return f"Headline: {article['headline']}\nSummary: {summary or '(none)'}"


def score_article(client: anthropic.Anthropic, article: dict) -> dict:
    """Ask Claude for {score, confidence, reason} for one article."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=256,
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": format_article(article)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"Unexpected stop_reason={response.stop_reason} for article {article['id']}")
    text = next(b.text for b in response.content if b.type == "text")
    result = json.loads(text)
    # Belt and braces: the schema constrains the range, but clamp anyway.
    result["score"] = max(-1.0, min(1.0, float(result["score"])))
    result["confidence"] = max(0.0, min(1.0, float(result["confidence"])))
    return result


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
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    scored: list[dict] = []
    for i, article in enumerate(raw["news"], start=1):
        result = score_article(client, article)
        scored.append({
            "id": article["id"],
            "created_at": article.get("created_at"),
            "headline": article["headline"],
            "source": article.get("source"),
            "url": article.get("url"),
            **result,
        })
        log.info("[%d/%d] %+.2f (conf %.2f) %s", i, len(raw["news"]), result["score"],
                 result["confidence"], article["headline"][:80])

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
