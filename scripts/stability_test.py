"""Repeated-run stability test: score identical input N times with the production scorer.

Input is taken from a saved scored file (NOT re-fetched from Alpaca, because headlines can
be edited after publication). Scored files do not store summaries, so every item is sent
as headline only; score_news.format_batch then writes "Summary: (none)". All runs see
byte-identical text, in the same order, in the same batches of score_news.BATCH_SIZE.

The scorer is imported unchanged. Token usage is recorded by wrapping the Anthropic SDK's
Messages.create from outside, so cost can be reported without editing score_news.py.

Outputs (in data/experiments/stability_<date>/):
  input.json          the exact items sent, with a SHA-256 of the rendered batch text
  run_<k>.json        per-run scores, plus usage for every API request (incl. retries)
  results.json        all computed statistics
  results.md          human-readable summary

Usage: uv run python scripts/stability_test.py --date 2026-10-26 --runs 5
Re-running resumes: completed run_<k>.json files are kept and only missing runs are scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics as st
import sys
import threading
import time
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
import score_news  # noqa: E402  (the production scorer, unmodified)
from aggregate import BEARISH_MAX, BULLISH_MIN, label_for  # noqa: E402

log = logging.getLogger("stability_test")

CONF_CUTOFF = 0.3
MAX_BATCH_ATTEMPTS = 3  # a batch that raises (refusal / bad idx) is retried, and logged
# Claude Sonnet 5.5 list prices, USD per million tokens (Anthropic API, 2026-09-25 table)
PRICE_PER_MTOK = {"input": 2.00, "output": 10.00, "cache_write_5m": 2.50, "cache_read": 0.20}

_usage_lock = threading.Lock()
_usage_log: list[dict] = []


def install_usage_recorder() -> None:
    """Wrap Messages.create so every response's usage is appended to _usage_log."""
    from anthropic.resources.messages import Messages

    original = Messages.create

    def recording_create(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        u = response.usage
        with _usage_lock:
            _usage_log.append({
                "request_id": getattr(response, "_request_id", None),
                "model": response.model,
                "stop_reason": response.stop_reason,
                "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens,
                "cache_creation_input_tokens": u.cache_creation_input_tokens or 0,
                "cache_read_input_tokens": u.cache_read_input_tokens or 0,
            })
        return response

    Messages.create = recording_create


def usage_cost(usage: list[dict]) -> dict:
    tot = {k: sum(u[k] for u in usage) for k in
           ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    usd = (tot["input_tokens"] * PRICE_PER_MTOK["input"]
           + tot["output_tokens"] * PRICE_PER_MTOK["output"]
           + tot["cache_creation_input_tokens"] * PRICE_PER_MTOK["cache_write_5m"]
           + tot["cache_read_input_tokens"] * PRICE_PER_MTOK["cache_read"]) / 1e6
    return {"requests": len(usage), **tot, "usd": round(usd, 4)}


def load_input(day: date) -> list[dict]:
    """Headline-only items, in the saved file's order. Summaries are not stored there."""
    record = json.loads((score_news.SCORED_DIR / f"scored_{day.isoformat()}.json").read_text())
    items = [{"id": r["id"], "headline": r["headline"]} for r in record["scored"]]
    if len({it["id"] for it in items}) != len(items):
        raise SystemExit("Duplicate article ids in the scored file; cannot key runs by id")
    return items, record


def batches(items: list[dict]) -> list[list[dict]]:
    n = score_news.BATCH_SIZE
    return [items[i:i + n] for i in range(0, len(items), n)]


def score_run(items: list[dict], k: int, out_dir: Path) -> None:
    """Score all items once with the production scorer and write run_<k>.json."""
    score_batch = score_news.make_anthropic_scorer()
    start_idx = len(_usage_log)
    rows, failures = [], []
    bs = batches(items)
    t0 = time.monotonic()
    for b, batch in enumerate(bs, start=1):
        for attempt in range(1, MAX_BATCH_ATTEMPTS + 1):
            try:
                results = score_batch(batch)
                break
            except RuntimeError as e:  # refusal, max_tokens, or idx mismatch
                failures.append({"batch": b, "attempt": attempt, "error": str(e)})
                log.warning("run %d batch %d attempt %d failed: %s", k, b, attempt, e)
                if attempt == MAX_BATCH_ATTEMPTS:
                    raise
        for it, res in zip(batch, results):
            rows.append({"id": it["id"], **res})
        if b % 10 == 0 or b == len(bs):
            log.info("run %d: batch %d/%d (%.0fs)", k, b, len(bs), time.monotonic() - t0)
    usage = _usage_log[start_idx:]
    out = {
        "run": k,
        "model": score_news.ANTHROPIC_MODEL,
        "prompt_version": score_news.PROMPT_VERSION,
        "batch_size": score_news.BATCH_SIZE,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.monotonic() - t0, 1),
        "batch_failures": failures,
        "usage": usage,
        "cost": usage_cost(usage),
        "scored": rows,
    }
    (out_dir / f"run_{k}.json").write_text(json.dumps(out, indent=2))
    log.info("run %d saved: %d rows, $%.4f", k, len(rows), out["cost"]["usd"])


def regime(rows: list[dict]) -> float | None:
    """Same formula as aggregate.py: confidence-weighted mean score."""
    w = sum(r["confidence"] for r in rows)
    return None if w == 0 else sum(r["score"] * r["confidence"] for r in rows) / w


def analyse(items: list[dict], runs: list[dict]) -> dict:
    ids = [it["id"] for it in items]
    headline = {it["id"]: it["headline"] for it in items}
    by_run = [{r["id"]: r for r in run["scored"]} for run in runs]
    for k, m in enumerate(by_run, start=1):
        if set(m) != set(ids):
            raise SystemExit(f"run {k} does not cover exactly the input ids")

    # 1. daily scores per run
    per_run = []
    for run, m in zip(runs, by_run):
        rows = list(m.values())
        hi = [r for r in rows if r["confidence"] >= CONF_CUTOFF]
        s_all, s_hi = regime(rows), regime(hi)
        per_run.append({
            "run": run["run"],
            "regime_all": s_all, "label_all": label_for(s_all),
            "n_conf_ge_cutoff": len(hi),
            "regime_conf_ge_cutoff": s_hi,
            "label_conf_ge_cutoff": None if s_hi is None else label_for(s_hi),
        })

    # 2. across runs
    def spread(key):
        v = [p[key] for p in per_run if p[key] is not None]
        return {"min": min(v), "max": max(v), "range": max(v) - min(v), "std": st.stdev(v)}

    across = {
        "regime_all": spread("regime_all"),
        "regime_conf_ge_cutoff": spread("regime_conf_ge_cutoff"),
        "label_all_values": sorted({p["label_all"] for p in per_run}),
        "label_conf_ge_cutoff_values": sorted({p["label_conf_ge_cutoff"] for p in per_run}),
    }
    across["label_all_changed"] = len(across["label_all_values"]) > 1
    across["label_conf_ge_cutoff_changed"] = len(across["label_conf_ge_cutoff_values"]) > 1

    # 3 & 4. per headline
    per_item = []
    for i in ids:
        scores = [m[i]["score"] for m in by_run]
        confs = [m[i]["confidence"] for m in by_run]
        labels = [label_for(s) for s in scores]
        per_item.append({
            "id": i, "headline": headline[i], "scores": scores, "confidences": confs,
            "labels": labels,
            "score_std": st.stdev(scores), "conf_std": st.stdev(confs),
            "score_range": max(scores) - min(scores),
            "score_identical": len(set(scores)) == 1,
            "conf_identical": len(set(confs)) == 1,
            "label_changed": len(set(labels)) > 1,
        })
    n = len(per_item)
    movers = sorted(per_item, key=lambda r: (r["score_range"], r["score_std"]), reverse=True)[:5]
    headline_stats = {
        "n_headlines": n,
        "mean_score_std": st.fmean(r["score_std"] for r in per_item),
        "mean_conf_std": st.fmean(r["conf_std"] for r in per_item),
        "share_score_identical_all_runs": sum(r["score_identical"] for r in per_item) / n,
        "share_conf_identical_all_runs": sum(r["conf_identical"] for r in per_item) / n,
        "share_label_changed": sum(r["label_changed"] for r in per_item) / n,
        "n_label_changed": sum(r["label_changed"] for r in per_item),
        "n_conf_ge_cutoff_all_runs": sum(all(c >= CONF_CUTOFF for c in r["confidences"]) for r in per_item),
        "n_conf_ge_cutoff_some_runs": sum(any(c >= CONF_CUTOFF for c in r["confidences"])
                                          and not all(c >= CONF_CUTOFF for c in r["confidences"])
                                          for r in per_item),
        "top5_movers": [{k: r[k] for k in ("id", "headline", "scores", "confidences", "labels",
                                           "score_range", "score_std")} for r in movers],
    }

    # 5. cost
    all_usage = [u for run in runs for u in run["usage"]]
    cost = {"per_run": [run["cost"] for run in runs], "total": usage_cost(all_usage),
            "prices_usd_per_mtok": PRICE_PER_MTOK,
            "batch_failures": [f | {"run": run["run"]} for run in runs for f in run["batch_failures"]]}

    return {"per_run": per_run, "across_runs": across, "per_headline": headline_stats,
            "cost": cost, "per_item": per_item}


def fmt(x, nd=4):
    return "n/a" if x is None else f"{x:+.{nd}f}"


def write_markdown(res: dict, meta: dict, path: Path) -> None:
    a, h, c = res["across_runs"], res["per_headline"], res["cost"]
    L = [f"# Stability test: {meta['source_date']}, {meta['runs']} runs", "",
         f"- Model `{meta['model']}`, prompt `{meta['prompt_version']}`, batch size {meta['batch_size']}, "
         f"{h['n_headlines']} headlines in the saved file's order.",
         "- Input: headlines only from the saved scored file (summaries were not stored), so every "
         "item was sent with `Summary: (none)`. Not comparable to the production scores for this date.",
         f"- Input SHA-256 (rendered batch text): `{meta['input_sha256']}`",
         f"- Daily score = confidence-weighted mean score (as in aggregate.py). Labels: bearish < "
         f"{BEARISH_MAX}, bullish > {BULLISH_MIN}. Standard deviations are sample (n-1).", "",
         "## 1. Daily score per run", "",
         f"| Run | Score (all) | Label | Score (conf >= {CONF_CUTOFF}) | n | Label |",
         "|---|---|---|---|---|---|"]
    for p in res["per_run"]:
        L.append(f"| {p['run']} | {fmt(p['regime_all'])} | {p['label_all']} | "
                 f"{fmt(p['regime_conf_ge_cutoff'])} | {p['n_conf_ge_cutoff']} | {p['label_conf_ge_cutoff']} |")
    L += ["", "## 2. Across runs", "",
          "| Daily score | Min | Max | Range | Std | Label changed? |", "|---|---|---|---|---|---|"]
    for key, name, lab in (("regime_all", "All headlines", "label_all"),
                           ("regime_conf_ge_cutoff", f"Conf >= {CONF_CUTOFF}", "label_conf_ge_cutoff")):
        s = a[key]
        L.append(f"| {name} | {fmt(s['min'])} | {fmt(s['max'])} | {s['range']:.4f} | {s['std']:.4f} | "
                 f"{'yes' if a[lab + '_changed'] else 'no'} ({', '.join(a[lab + '_values'])}) |")
    L += ["", "## 3. Per headline", "",
          "| Metric | Value |", "|---|---|",
          f"| Mean std of score across runs | {h['mean_score_std']:.4f} |",
          f"| Mean std of confidence across runs | {h['mean_conf_std']:.4f} |",
          f"| Headlines with identical score in all runs | {h['share_score_identical_all_runs']:.1%} |",
          f"| Headlines with identical confidence in all runs | {h['share_conf_identical_all_runs']:.1%} |",
          "", "## 4. Label flips", "",
          f"{h['share_label_changed']:.1%} of headlines ({h['n_label_changed']} of {h['n_headlines']}) "
          "changed bearish/neutral/bullish label in at least one run.", "",
          "Top 5 headlines by score range across runs:", "",
          "| Headline | Scores (runs 1-5) | Range | Labels |", "|---|---|---|---|"]
    for r in h["top5_movers"]:
        L.append(f"| {r['headline'].replace('|', '/')} | {', '.join(f'{s:+.2f}' for s in r['scores'])} | "
                 f"{r['score_range']:.2f} | {', '.join(r['labels'])} |")
    t = c["total"]
    L += ["", "## 5. Cost", "",
          f"Total ${t['usd']:.4f} for {t['requests']} requests: {t['input_tokens']:,} input, "
          f"{t['output_tokens']:,} output, {t['cache_creation_input_tokens']:,} cache-write, "
          f"{t['cache_read_input_tokens']:,} cache-read tokens. Prices (USD/MTok): "
          + ", ".join(f"{k} {v}" for k, v in c["prices_usd_per_mtok"].items()) + ".",
          f"Batch retries: {len(c['batch_failures'])}.",
          "", "## Notes", "",
          f"- The conf >= {CONF_CUTOFF} score rests on very few headlines. {h['n_conf_ge_cutoff_all_runs']} "
          f"headlines cleared the cutoff in every run and {h['n_conf_ge_cutoff_some_runs']} crossed it in "
          "some runs only, so part of that score's run-to-run spread comes from the subset changing."]
    path.write_text("\n".join(L) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", type=date.fromisoformat, required=True)
    p.add_argument("--runs", type=int, default=5)
    args = p.parse_args()
    load_dotenv()

    out_dir = Path("data/experiments") / f"stability_{args.date.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    items, source = load_input(args.date)
    rendered = "\n\n=====\n\n".join(score_news.format_batch(b) for b in batches(items))
    meta = {
        "source_file": f"data/scored/scored_{args.date.isoformat()}.json",
        "source_date": source["date"], "source_model": source["model"],
        "model": score_news.ANTHROPIC_MODEL, "prompt_version": score_news.PROMPT_VERSION,
        "batch_size": score_news.BATCH_SIZE, "runs": args.runs, "conf_cutoff": CONF_CUTOFF,
        "n_items": len(items), "input_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "note": "headline only; summaries not stored in scored files",
    }
    (out_dir / "input.json").write_text(json.dumps({**meta, "items": items}, indent=2))

    install_usage_recorder()
    for k in range(1, args.runs + 1):
        if (out_dir / f"run_{k}.json").exists():
            log.info("run %d already saved; skipping", k)
            continue
        score_run(items, k, out_dir)

    runs = [json.loads((out_dir / f"run_{k}.json").read_text()) for k in range(1, args.runs + 1)]
    res = analyse(items, runs)
    (out_dir / "results.json").write_text(json.dumps({"meta": meta, **res}, indent=2))
    write_markdown(res, meta, out_dir / "results.md")
    log.info("Wrote %s and results.md", out_dir / "results.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    main()
