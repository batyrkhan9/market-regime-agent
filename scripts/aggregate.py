"""Collapse one day's scored headlines into a single signal row.

Input : data/scored/scored_YYYY-MM-DD.json
Output: appended row in data/signals.csv (existing rows are never modified)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger("aggregate")

SCORED_DIR = Path("data/scored")
SIGNALS_PATH = Path("data/signals.csv")
BEARISH_MAX = -0.2  # label is bearish if score < this
BULLISH_MIN = 0.2   # label is bullish if score > this
N_REASONS = 3
COLUMNS = ["date", "regime_score", "confidence", "label", "n_headlines",
           "top_reasons", "model", "prompt_version", "window_start", "window_end"]


def label_for(score: float) -> str:
    """Map a regime score to bearish / neutral / bullish."""
    if score < BEARISH_MAX:
        return "bearish"
    if score > BULLISH_MIN:
        return "bullish"
    return "neutral"


def aggregate(scored: list[dict]) -> tuple[float, float, str]:
    """Return (regime_score, confidence, top_reasons) for a list of scored headlines."""
    if not scored:
        return 0.0, 0.0, ""
    df = pd.DataFrame(scored)
    weight_sum = df["confidence"].sum()
    if weight_sum == 0:
        return 0.0, 0.0, ""
    regime_score = float((df["score"] * df["confidence"]).sum() / weight_sum)
    confidence = float(df["confidence"].mean())
    # "Top" = highest confidence * |score|: the headlines that moved the needle most.
    df["impact"] = df["confidence"] * df["score"].abs()
    top = df.sort_values("impact", ascending=False).head(N_REASONS)
    top_reasons = " | ".join(f"[{s:+.2f}] {r}" for s, r in zip(top["score"], top["reason"]))
    return round(regime_score, 4), round(confidence, 4), top_reasons


def load_signals() -> pd.DataFrame:
    """Read the signals CSV, or return an empty frame with the right columns."""
    if SIGNALS_PATH.exists():
        return pd.read_csv(SIGNALS_PATH, dtype={"date": str})
    return pd.DataFrame(columns=COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=date.today(),
                        help="Date of the scored file to aggregate (YYYY-MM-DD)")
    args = parser.parse_args()

    scored_path = SCORED_DIR / f"scored_{args.date.isoformat()}.json"
    if not scored_path.exists():
        raise SystemExit(f"Scored file not found: {scored_path} (run score_news.py first)")
    record = json.loads(scored_path.read_text())

    signals = load_signals()
    if record["date"] in set(signals["date"]):
        log.info("Row for %s already in %s; not overwriting", record["date"], SIGNALS_PATH)
        return

    regime_score, confidence, top_reasons = aggregate(record["scored"])
    row = {
        "date": record["date"],
        "regime_score": regime_score,
        "confidence": confidence,
        "label": label_for(regime_score),
        "n_headlines": len(record["scored"]),
        "top_reasons": top_reasons,
        "model": record["model"],
        "prompt_version": record["prompt_version"],
        "window_start": record["window_start"],
        "window_end": record["window_end"],
    }
    signals = pd.concat([signals, pd.DataFrame([row])], ignore_index=True)
    signals = signals.sort_values("date")[COLUMNS]
    SIGNALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(SIGNALS_PATH, index=False)
    log.info("Appended %s: score=%+.3f conf=%.2f label=%s n=%d",
             row["date"], regime_score, confidence, row["label"], row["n_headlines"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
