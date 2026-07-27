"""Regression tests for the codebase-review fixes.

Covers:
- backtest cost math (long-short must pay costs on BOTH legs; short leg is true short P&L)
- per-fold quantile assignment (no cross-fold lookahead in portfolio formation)
- previous_session strict-before semantics
- logging: every named logger gets a handler
- edgar_yf provider stays labeled synthetic
- CLI data-contract enforcement raises on malformed tables
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera
import pytest

from earnings_ml.backtest.portfolio import run_backtest
from earnings_ml.calendars import exchange as ex
from earnings_ml.config import load_config

D = pd.Timestamp("2020-01-14")


def _mini_events(n: int):
    return pd.DataFrame([{
        "event_id": f"E{i}", "security_id": f"S{i}", "exclusion_reason": "",
        "entry_session": D, "reaction_exit_session": D, "drift_exit_session": D,
    } for i in range(n)])


def _mini_prices(n: int):
    # gross reaction return for security i = i * 1%
    return pd.DataFrame([{
        "security_id": f"S{i}", "date": D, "open": 100.0,
        "high": 101.0, "low": 99.0, "close": 100.0 * (1 + 0.01 * i), "volume": 1e6,
    } for i in range(n)])


def test_long_short_pays_costs_on_both_legs():
    cfg = load_config()
    events, prices = _mini_events(10), _mini_prices(10)
    preds = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(10)],
        "prediction": [float(i) for i in range(10)],
        "fold": ["test_2020"] * 10,
        "test_start": [pd.Timestamp("2020-01-01")] * 10,
    })
    res = run_backtest(events, preds, prices, cfg, horizon="reaction")
    sweep = res["cost_sweep"]
    # top quintile = E8,E9 (gross 8%, 9%); bottom = E0,E1 (gross 0%, 1%)
    for c, block in sweep.items():
        c_dec = c / 10000.0
        assert block["long_only_net"] == pytest.approx(0.085 - c_dec, abs=1e-12)
        assert block["short_only_net"] == pytest.approx(-0.005 - c_dec, abs=1e-12)
        assert block["long_short_net"] == pytest.approx(0.080 - 2 * c_dec, abs=1e-12)
    ls_by_cost = [sweep[c]["long_short_net"] for c in sorted(sweep)]
    assert all(b < a for a, b in zip(ls_by_cost, ls_by_cost[1:], strict=False)), (
        "long-short net must strictly decrease as costs rise"
    )


def test_quantiles_assigned_per_fold_not_pooled():
    cfg = load_config()
    events, prices = _mini_events(10), _mini_prices(10)
    # fold A predictions (1..5) are all below fold B predictions (101..105): pooled
    # assignment would starve fold A of top-quantile membership entirely.
    preds = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(10)],
        "prediction": [1, 2, 3, 4, 5, 101, 102, 103, 104, 105],
        "fold": ["test_2020"] * 5 + ["test_2021"] * 5,
        "test_start": [pd.Timestamp("2020-01-01")] * 5 + [pd.Timestamp("2021-01-01")] * 5,
    })
    res = run_backtest(events, preds, prices, cfg, horizon="reaction")
    # per-fold top quintiles: E4 (gross 4%) and E9 (gross 9%) -> mean 6.5%
    assert res["cost_sweep"][0]["long_only_gross"] == pytest.approx(0.065, abs=1e-12)
    assert res["cost_sweep"][0]["n_long"] == 2
    # per-year stats require test_start to be merged through
    assert set(res["primary"]["long_by_year"].keys()) == {2020, 2021}


def test_previous_session_is_strictly_before():
    wed = pd.Timestamp("2020-01-29")  # a session
    prev = ex.previous_session(wed)
    assert prev == pd.Timestamp("2020-01-28")
    assert prev < wed
    sun = pd.Timestamp("2020-02-02")  # Sunday
    assert ex.previous_session(sun) == pd.Timestamp("2020-01-31")


def test_every_named_logger_gets_a_handler():
    from earnings_ml.logging import get_logger
    l1 = get_logger("earnings_ml.review_first")
    l2 = get_logger("earnings_ml.review_second")
    assert l1.handlers, "first logger must have a handler"
    assert l2.handlers, "later loggers must also have handlers (messages were silently dropped)"
    assert not l2.propagate


def test_edgar_yf_provider_stays_labeled_synthetic():
    from earnings_ml.ingestion.real import EdgarYFinanceProvider
    assert EdgarYFinanceProvider.is_synthetic is True, (
        "edgar_yf uses mock events/estimates/filings; it must keep the SYNTHETIC label"
    )


def test_cli_contract_validation_raises_on_malformed_table():
    from earnings_ml.cli import _validate_contract
    bad = pd.DataFrame([{"event_id": "E1", "security_id": "S1"}])  # missing required columns
    with pytest.raises(pandera.errors.SchemaError):
        _validate_contract("events", bad)


def test_long_by_year_present_in_primary_block():
    cfg = load_config()
    events, prices = _mini_events(10), _mini_prices(10)
    preds = pd.DataFrame({
        "event_id": [f"E{i}" for i in range(10)],
        "prediction": [float(i) for i in range(10)],
        "fold": ["test_2020"] * 10,
        "test_start": [pd.Timestamp("2020-01-01")] * 10,
    })
    res = run_backtest(events, preds, prices, cfg, horizon="reaction")
    assert "long_by_year" in res["primary"]
    assert np.isfinite(res["primary"]["long_short_annualized"])
