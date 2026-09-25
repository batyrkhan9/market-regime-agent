"""Run the full daily pipeline: fetch news -> score -> aggregate -> market regime.

Each step is a separate subprocess so that a failure in one stops the run with a
non-zero exit code and the failing step's name in the log.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

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
    parser.add_argument("--skip-market", action="store_true",
                        help="Skip the SPY download (useful for backfilling news only)")
    args = parser.parse_args()
    day = args.date.isoformat()

    log.info("Daily pipeline for %s (now %s)", day, datetime.now(ET).isoformat(timespec="seconds"))
    run_step("fetch_news", ["--date", day])
    run_step("score_news", ["--date", day])
    run_step("aggregate", ["--date", day])
    if not args.skip_market:
        run_step("market_regime", [])
    log.info("Pipeline complete for %s", day)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
