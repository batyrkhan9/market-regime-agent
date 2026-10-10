"""NYSE trading-day helpers shared by the pipeline scripts.

A trading day is a weekday that is not an NYSE holiday. The `holidays` package carries the
NYSE calendar, including observed holidays and one-off closures (e.g. days of mourning).
"""

from __future__ import annotations

from datetime import date, timedelta

import holidays

_NYSE = holidays.NYSE()  # lazily populates years as they are looked up


def is_trading_day(d: date) -> bool:
    """True if the NYSE is open on `d` (Mon-Fri and not a holiday)."""
    return d.weekday() < 5 and d not in _NYSE


def previous_trading_day(d: date) -> date:
    """Return the most recent trading day strictly before `d`."""
    d -= timedelta(days=1)
    while not is_trading_day(d):
        d -= timedelta(days=1)
    return d
