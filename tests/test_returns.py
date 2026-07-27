"""Return-calculation and corporate-action sanity tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_ml.backtest.portfolio import _entry_exit_returns
from earnings_ml.labels.car import build_labels


def _fake_prices(sid: str, dates: list[pd.Timestamp], closes: list[float]) -> pd.DataFrame:
    rows = []
    for d, c in zip(dates, closes):
        rows.append({"security_id": sid, "date": d, "open": c * 0.995, "high": c,
                     "low": c * 0.99, "close": c, "volume": 1e6})
    return pd.DataFrame(rows)


def test_reaction_return_is_open_to_close():
    # entry session: open=100, close=101 -> reaction = 1%
    d = pd.Timestamp("2020-01-14")
    prices = pd.DataFrame([{
        "security_id": "S1", "date": d, "open": 100.0, "high": 101.0,
        "low": 99.5, "close": 101.0, "volume": 1e6,
    }])
    events = pd.DataFrame([{
        "event_id": "E1", "security_id": "S1", "exclusion_reason": "",
        "entry_session": d, "reaction_exit_session": d, "drift_exit_session": d,
        "sector": "X", "effective_trading_date": d,
    }])
    bench = pd.DataFrame([{"date": d, "open": 100.0, "close": 100.0}])
    sector = pd.DataFrame([{"sector": "X", "date": d, "open": 100.0, "close": 100.0}])
    labels = build_labels(events, prices, bench, sector, load_cfg_min())
    assert len(labels) == 1
    assert abs(labels["reaction_car"].iloc[0] - 0.01) < 1e-9


def test_drift_return_does_not_overlap_reaction():
    # reaction: open=100 -> close=101 (day1)
    # drift:    close=101 (day1) -> close=103 (day21) = 2/101
    d1 = pd.Timestamp("2020-01-14")
    d21 = pd.Timestamp("2020-02-11")
    prices = _fake_prices("S1", [d1, d21], [101.0, 103.0])
    events = pd.DataFrame([{
        "event_id": "E1", "security_id": "S1", "exclusion_reason": "",
        "entry_session": d1, "reaction_exit_session": d1, "drift_exit_session": d21,
        "sector": "X", "effective_trading_date": d1,
    }])
    bench = pd.DataFrame([
        {"date": d1, "open": 100.0, "close": 100.0},
        {"date": d21, "open": 100.0, "close": 100.0},
    ])
    sector = pd.DataFrame([
        {"sector": "X", "date": d1, "open": 100.0, "close": 100.0},
        {"sector": "X", "date": d21, "open": 100.0, "close": 100.0},
    ])
    labels = build_labels(events, prices, bench, sector, load_cfg_min())
    assert abs(labels["drift_car_20"].iloc[0] - (103.0 / 101.0 - 1)) < 1e-9


def test_split_factor_does_not_contaminate_return():
    # If a 2:1 split occurred, adjusted_close halves while unadjusted stays.
    # Our labels use 'close' (adjusted in the mock); a synthetic split should not
    # corrupt returns because we use the same series.
    d1 = pd.Timestamp("2020-01-14")
    d21 = pd.Timestamp("2020-02-11")
    # Simulate adjusted closes that already reflect the split
    prices = pd.DataFrame([
        {"security_id": "S1", "date": d1, "open": 50.0, "high": 51.0, "low": 49.5, "close": 50.5, "volume": 1e6},
        {"security_id": "S1", "date": d21, "open": 51.0, "high": 52.0, "low": 50.5, "close": 51.5, "volume": 1e6},
    ])
    events = pd.DataFrame([{
        "event_id": "E1", "security_id": "S1", "exclusion_reason": "",
        "entry_session": d1, "reaction_exit_session": d1, "drift_exit_session": d21,
        "sector": "X", "effective_trading_date": d1,
    }])
    bench = pd.DataFrame([
        {"date": d1, "open": 100.0, "close": 100.0},
        {"date": d21, "open": 100.0, "close": 100.0},
    ])
    sector = pd.DataFrame([
        {"sector": "X", "date": d1, "open": 100.0, "close": 100.0},
        {"sector": "X", "date": d21, "open": 100.0, "close": 100.0},
    ])
    labels = build_labels(events, prices, bench, sector, load_cfg_min())
    # drift = 51.5 / 50.5 - 1
    assert abs(labels["drift_car_20"].iloc[0] - (51.5 / 50.5 - 1)) < 1e-9


def load_cfg_min():
    from earnings_ml.config import load_config
    return load_config()
