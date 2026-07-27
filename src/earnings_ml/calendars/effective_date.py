"""Effective-trading-date rules.

Rules (from spec section 7):
- before_market_open (BMO): effective = same trading session.
- after_market_close (AMC):  effective = next trading session.
- weekend/holiday release:    effective = next trading session.
- intraday:                   excluded from primary (exclusion_reason set).
- unknown:                    excluded from primary (sensitivity-only).

The prediction timestamp is `effective_date 09:35 ET` and the order-submission cutoff is
`effective_date 09:25 ET` (features must be strictly available before 09:25 ET).
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from . import exchange as ex

TZ_ET = "America/New_York"
ORDER_CUTOFF_ET = "09:25"
PREDICTION_TS_ET = "09:35"

VALID_TIMING = {"before_market_open", "after_market_close", "intraday", "unknown"}
EXCLUDED_TIMING = {"intraday", "unknown"}


def _to_et(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(tz) if t.tzinfo is None else t.tz_convert(tz)


def effective_trading_date(
    announcement_ts: pd.Timestamp,
    timing_class: str,
    tz: str = TZ_ET,
    exchange: str = "XNYS",
) -> pd.Timestamp:
    """Return the trading session at which the information first becomes tradable."""
    if timing_class not in VALID_TIMING:
        raise ValueError(f"invalid timing_class: {timing_class!r}")

    ann_et = _to_et(announcement_ts, tz)
    # Use the ET-local calendar date (not UTC date) for session logic — an AMC release
    # at 21:05 ET on Jan 28 is still "Jan 28" for market-calendar purposes.
    ann_date = ann_et.normalize().replace(tzinfo=None)

    if ex.is_session(ann_date, exchange):
        if timing_class == "before_market_open":
            return ann_date
        if timing_class == "after_market_close":
            return ex.next_session(ann_date, exchange)
        # intraday / unknown -> next session (then excluded upstream)
        return ex.next_session(ann_date, exchange)
    # weekend/holiday release
    return ex.next_session(ann_date, exchange)


def prediction_timestamp(effective_date: pd.Timestamp, cfg: Config | None = None) -> pd.Timestamp:
    tz = cfg.tz if cfg else TZ_ET
    ts_str = cfg.base.get("prediction_timestamp_et", PREDICTION_TS_ET) if cfg else PREDICTION_TS_ET
    eff = pd.Timestamp(effective_date).tz_localize(tz)
    hh, mm = (int(x) for x in ts_str.split(":"))
    return eff + pd.Timedelta(hours=hh, minutes=mm)


def order_submission_cutoff(effective_date: pd.Timestamp, cfg: Config | None = None) -> pd.Timestamp:
    tz = cfg.tz if cfg else TZ_ET
    ts_str = cfg.base.get("order_submission_cutoff_et", ORDER_CUTOFF_ET) if cfg else ORDER_CUTOFF_ET
    eff = pd.Timestamp(effective_date).tz_localize(tz)
    hh, mm = (int(x) for x in ts_str.split(":"))
    return eff + pd.Timedelta(hours=hh, minutes=mm)


def should_exclude(timing_class: str) -> tuple[bool, str]:
    if timing_class in EXCLUDED_TIMING:
        return True, f"timing_class={timing_class} excluded from primary"
    return False, ""
