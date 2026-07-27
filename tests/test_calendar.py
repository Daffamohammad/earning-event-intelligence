"""Calendar / effective-date tests against REAL historical announcements.

These are the highest-value tests: they verify the XNYS calendar logic against known
facts (Apple AMC Jan 28 2020 -> effective Jan 29; JPM BMO Jan 14 2020 -> effective Jan 14).
"""
from __future__ import annotations

import pandas as pd

from earnings_ml.calendars import exchange as ex
from earnings_ml.calendars.effective_date import effective_trading_date, order_submission_cutoff, prediction_timestamp
from earnings_ml.validation.golden import evaluate_golden


def test_weekend_release_uses_next_session():
    # Friday Jan 31 2020 was a session; Saturday Feb 1 is not.
    sat = pd.Timestamp("2020-02-01 10:00:00").tz_localize("America/New_York")
    eff = effective_trading_date(sat, "after_market_close", tz="America/New_York")
    assert ex.is_session(eff)
    assert eff == pd.Timestamp("2020-02-03")  # Monday


def test_bmo_same_session():
    bmo = pd.Timestamp("2020-01-14 08:00:00").tz_localize("America/New_York")
    eff = effective_trading_date(bmo, "before_market_open", tz="America/New_York")
    assert eff == pd.Timestamp("2020-01-14")


def test_amc_next_session():
    amc = pd.Timestamp("2020-01-28 16:30:00").tz_localize("America/New_York")
    eff = effective_trading_date(amc, "after_market_close", tz="America/New_York")
    assert eff == pd.Timestamp("2020-01-29")


def test_holiday_release_uses_next_session():
    # 2020-09-07 was Labor Day (XNYS closed)
    labor = pd.Timestamp("2020-09-07 08:00:00").tz_localize("America/New_York")
    eff = effective_trading_date(labor, "before_market_open", tz="America/New_York")
    assert eff == pd.Timestamp("2020-09-08")


def test_intraday_excluded_from_primary():
    intra = pd.Timestamp("2020-03-04 12:00:00").tz_localize("America/New_York")
    eff = effective_trading_date(intra, "intraday", tz="America/New_York")
    assert eff == pd.Timestamp("2020-03-05")  # next session, but upstream excludes


def test_prediction_timestamp_after_order_cutoff():
    eff = pd.Timestamp("2020-01-29")
    pt = prediction_timestamp(eff)
    oc = order_submission_cutoff(eff)
    assert pt > oc
    assert pt == pd.Timestamp("2020-01-29 09:35:00-05:00")
    assert oc == pd.Timestamp("2020-01-29 09:25:00-05:00")


def test_shift_sessions_roundtrip():
    d = pd.Timestamp("2020-01-29")
    forward = ex.shift_sessions(d, 20)
    back = ex.shift_sessions(forward, -20)
    assert back == d


def test_shift_sessions_always_returns_session():
    """Per Checkpoint B #3: shift_sessions(entry, ±n) must return a real session."""
    entry = pd.Timestamp("2020-01-29")
    for n in range(-20, 21):
        result = ex.shift_sessions(entry, n)
        assert ex.is_session(result), f"shift_sessions({entry}, {n})={result} not a session"


def test_golden_effective_date_accuracy_meets_threshold(cfg):
    r = evaluate_golden(cfg)
    assert r["status"] == "ok", "golden dataset must exist"
    assert r["n"] >= 50, "need at least 50 hand-verified events"
    assert r["effective_date_accuracy"] >= 0.98, "effective-date accuracy must be >= 98%"
    assert r["duplicate_rate"] < 0.005
    assert r["timing_class_valid_rate"] == 1.0


def test_all_timestamps_are_tz_aware(cfg):
    """Hard guard: every prediction_timestamp / order_submission_cutoff is tz-aware."""
    from earnings_ml.calendars.effective_date import order_submission_cutoff, prediction_timestamp
    for d in [pd.Timestamp("2020-01-29"), pd.Timestamp("2021-07-15")]:
        assert prediction_timestamp(d, cfg).tzinfo is not None
        assert order_submission_cutoff(d, cfg).tzinfo is not None
