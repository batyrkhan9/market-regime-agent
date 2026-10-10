"""Fetch financial news from the Alpaca News API for one trading-day window.

Window: previous NYSE trading day 16:00 ET -> target date 16:00 ET (capped at "now").
So Monday's window starts Friday 16:00 ET, and the day after a holiday starts at the
close before the holiday: weekend and holiday news lands in the next trading day's row.
Output: data/raw/news_YYYY-MM-DD.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from trading_days import previous_trading_day
from urllib3.util import Retry

log = logging.getLogger("fetch_news")

NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
ET = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)
PAGE_SIZE = 50  # Alpaca's maximum per request
RAW_DIR = Path("data/raw")


def news_window(target: date) -> tuple[datetime, datetime]:
    """Return (start, end) datetimes in ET for the news window ending on `target`."""
    start = datetime.combine(previous_trading_day(target), MARKET_CLOSE, tzinfo=ET)
    end = datetime.combine(target, MARKET_CLOSE, tzinfo=ET)
    now = datetime.now(ET)
    if end > now:
        log.warning("Window end %s is in the future; capping at now (%s)", end, now)
        end = now
    return start, end


def make_session(key: str, secret: str) -> requests.Session:
    """Build a requests session with auth headers and retries for 429/5xx."""
    session = requests.Session()
    session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def fetch_news(session: requests.Session, start: datetime, end: datetime, max_items: int) -> list[dict]:
    """Page through the Alpaca News API and return de-duplicated articles."""
    params = {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "limit": PAGE_SIZE,
        "sort": "asc",
        "include_content": "false",
    }
    seen: dict[str, dict] = {}
    while True:
        resp = session.get(NEWS_URL, params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("news", []):
            seen[str(item["id"])] = item
        log.info("Fetched page, total articles so far: %d", len(seen))
        token = payload.get("next_page_token")
        if not token or len(seen) >= max_items:
            break
        params["page_token"] = token
    if len(seen) >= max_items:
        log.warning("Hit max_items=%d; window may be truncated", max_items)
    return list(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(ET).date(),
                        help="Target date YYYY-MM-DD (default: today in ET)")
    parser.add_argument("--max-items", type=int, default=1000, help="Safety cap on articles")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing raw file")
    args = parser.parse_args()

    load_dotenv()
    key, secret = os.getenv("ALPACA_KEY"), os.getenv("ALPACA_SECRET")
    if not key or not secret:
        raise SystemExit("ALPACA_KEY / ALPACA_SECRET missing from environment (.env)")

    out_path = RAW_DIR / f"news_{args.date.isoformat()}.json"
    if out_path.exists() and not args.force:
        log.info("%s already exists; skipping (use --force to refetch)", out_path)
        return

    start, end = news_window(args.date)
    log.info("Fetching news from %s to %s", start, end)
    articles = fetch_news(make_session(key, secret), start, end, args.max_items)

    record = {
        "date": args.date.isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "fetched_at": datetime.now(ET).isoformat(),
        "n_articles": len(articles),
        "news": articles,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2))
    log.info("Saved %d articles to %s", len(articles), out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
