"""Build a price-based market regime label from SPY, for testing the news signal against.

Output: data/market_regime.csv (fully recomputed each run; it is derived from public prices).
"""

from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger("market_regime")

TICKER = "SPY"
OUT_PATH = Path("data/market_regime.csv")
WINDOW = 20                  # trading days, roughly one calendar month
RET_BULLISH = 0.02           # 20-day return above +2% -> bullish
RET_BEARISH = -0.02          # 20-day return below -2% -> bearish
VOL_STRESS = 0.25            # annualised 20-day vol above 25% -> bearish regardless of return
TRADING_DAYS_PER_YEAR = 252


def download_prices(start: date) -> pd.DataFrame:
    """Return a DataFrame with a 'close' column indexed by calendar date."""
    hist = yf.Ticker(TICKER).history(start=start.isoformat(), auto_adjust=True)
    if hist.empty:
        raise RuntimeError(f"yfinance returned no rows for {TICKER} from {start}")
    df = hist[["Close"]].rename(columns={"Close": "close"})
    df.index = pd.to_datetime(df.index).tz_localize(None).date
    df.index.name = "date"
    return df


def add_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling return, rolling volatility, and a three-way regime label."""
    df = df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_20d"] = df["close"].pct_change(WINDOW)
    df["vol_20d"] = df["ret_1d"].rolling(WINDOW).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    df["market_label"] = "neutral"
    df.loc[df["ret_20d"] > RET_BULLISH, "market_label"] = "bullish"
    df.loc[df["ret_20d"] < RET_BEARISH, "market_label"] = "bearish"
    df.loc[df["vol_20d"] > VOL_STRESS, "market_label"] = "bearish"
    df.loc[df["ret_20d"].isna() | df["vol_20d"].isna(), "market_label"] = ""  # warm-up rows
    return df.round({"close": 2, "ret_1d": 5, "ret_20d": 5, "vol_20d": 5})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat,
                        default=date.today() - timedelta(days=3 * 365),
                        help="First date to download (default: 3 years ago)")
    args = parser.parse_args()

    log.info("Downloading %s from %s", TICKER, args.start)
    prices = download_prices(args.start)
    regime = add_regime(prices)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    regime.to_csv(OUT_PATH)
    last = regime.iloc[-1]
    log.info("Saved %d rows to %s; latest %s: ret_20d=%+.2f%% vol_20d=%.1f%% label=%s",
             len(regime), OUT_PATH, regime.index[-1], 100 * last["ret_20d"],
             100 * last["vol_20d"], last["market_label"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
