"""Run the full daily pipeline: fetch news -> score -> aggregate -> market regime.

Each step is a separate subprocess so that a failure in one stops the run with a
non-zero exit code and the failing step's name in the log.

Rows are only produced for NYSE trading days. On a weekend or holiday the run exits 0
without doing anything, unless --mode test is given. Mode is recorded in signals.csv:
live (default when the date is today), backfill (default for past dates), test.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

from scripts.trading_days import is_trading_day

log = logging.getLogger("run_daily")

ET = ZoneInfo("America/New_York")
SCRIPTS = "scripts"


def run_step(name: str, args: list[str]) -> None:
    """Run one script with the current interpreter; raise if it exits non-zero."""
    cmd = [sys.executable, f"{SCRIPTS}/{name}.py", *args]
    log.info("=== START %s: %s", name, " ".join(cmd))
    t0 = time.monotonic()
    result = subprocess.run(cmd)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        log.error("=== FAILED %s after %.1fs (exit code %d)", name, elapsed, result.returncode)
        raise SystemExit(f"Pipeline stopped at step '{name}'")
    log.info("=== DONE %s in %.1fs", name, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date(),
                        help="Signal date YYYY-MM-DD (default: today in ET)")
    parser.add_argument("--mode", choices=["live", "backfill", "test"],
                        help="Row mode (default: live if --date is today in ET, else backfill)")
    parser.add_argument("--skip-market", action="store_true",
                        help="Skip the SPY download (useful for backfilling news only)")
    args = parser.parse_args()
    day = args.date.isoformat()
    today = datetime.now(ET).date()
    mode = args.mode or ("live" if args.date == today else "backfill")

    log.info("Daily pipeline for %s mode=%s (now %s)", day, mode,
             datetime.now(ET).isoformat(timespec="seconds"))
    if not is_trading_day(args.date):
        if mode != "test":
            log.info("%s is not an NYSE trading day; nothing to do (use --mode test to force)", day)
            return
        log.warning("%s is not an NYSE trading day; continuing because mode=test", day)
    run_step("fetch_news", ["--date", day])
    run_step("score_news", ["--date", day])
    run_step("aggregate", ["--date", day, "--mode", mode])
    if not args.skip_market:
        run_step("market_regime", [])
    log.info("Pipeline complete for %s", day)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
