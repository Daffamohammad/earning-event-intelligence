"""Exchange-calendar wrapper (XNYS by default).

All event-date logic MUST go through this module; never use plain calendar-day arithmetic.
"""
from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd

from ..config import Config


@lru_cache(maxsize=8)
def get_calendar(exchange: str = "XNYS") -> xcals.ExchangeCalendar:
    return xcals.get_calendar(exchange)


def sessions(cfg: Config | None = None, exchange: str | None = None) -> pd.DatetimeIndex:
    cal = get_calendar(exchange or (cfg.base.get("calendar", {}).get("exchange", "XNYS") if cfg else "XNYS"))
    start = pd.Timestamp("2010-01-01")
    end = pd.Timestamp.utcnow().tz_localize(None).floor("D") + pd.Timedelta(days=2)
    return cal.sessions_in_range(start, end)


def _naive_date(date: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(date)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.normalize()


def is_session(date: pd.Timestamp, exchange: str = "XNYS") -> bool:
    cal = get_calendar(exchange)
    return cal.is_session(_naive_date(date))


def next_session(date: pd.Timestamp, exchange: str = "XNYS") -> pd.Timestamp:
    """Return the first trading session STRICTLY AFTER `date`.

    If `date` is itself a session, that session is NOT returned (we want the next one).
    """
    cal = get_calendar(exchange)
    d = _naive_date(date)
    if cal.is_session(d):
        d = d + pd.Timedelta(days=1)
    nxt = cal.date_to_session(d, "next")
    return pd.Timestamp(nxt).normalize()


def previous_session(date: pd.Timestamp, exchange: str = "XNYS") -> pd.Timestamp:
    """Return the last trading session STRICTLY BEFORE `date`.

    If `date` is itself a session, that session is NOT returned (we want the prior one),
    mirroring the strict-after semantics of `next_session`.
    """
    cal = get_calendar(exchange)
    d = _naive_date(date)
    if cal.is_session(d):
        d = d - pd.Timedelta(days=1)
    prev = cal.date_to_session(d, direction="previous")
    return pd.Timestamp(prev).normalize()


def shift_sessions(date: pd.Timestamp, n: int, exchange: str = "XNYS") -> pd.Timestamp:
    """Shift by n trading sessions (n positive => forward in time)."""
    cal = get_calendar(exchange)
    end = pd.Timestamp.utcnow().tz_localize(None).floor("D") + pd.Timedelta(days=10)
    sessions_idx = cal.sessions_in_range("2010-01-01", end)
    target = _naive_date(date)
    pos = sessions_idx.searchsorted(target)
    if pos >= len(sessions_idx):
        pos = len(sessions_idx) - 1
    if pos > 0 and (sessions_idx[pos] != target) and abs(sessions_idx[pos - 1] - target) < abs(sessions_idx[pos] - target):
        # snap to nearest
        pos = pos - 1
    return pd.Timestamp(sessions_idx[max(0, min(pos + n, len(sessions_idx) - 1))]).normalize()


def nth_session_after(date: pd.Timestamp, n: int, exchange: str = "XNYS") -> pd.Timestamp:
    """The trading session that is n sessions strictly after `date` (n>=0)."""
    return shift_sessions(date, n, exchange)


def filter_sessions(dates: Iterable[pd.Timestamp], exchange: str = "XNYS") -> list[pd.Timestamp]:
    cal = get_calendar(exchange)
    return [d for d in dates if cal.is_session(pd.Timestamp(d).normalize())]
