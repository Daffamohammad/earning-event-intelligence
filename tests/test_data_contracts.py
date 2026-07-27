"""Data-contract tests: pandera schemas reject malformed inputs."""
from __future__ import annotations

import pandas as pd
import pytest

from data_contracts.schemas import SCHEMAS


def test_events_schema_rejects_bad_timing_class():
    bad = pd.DataFrame([{
        "event_id": "E1", "security_id": "S1", "ticker": "T",
        "announcement_timestamp": pd.Timestamp("2020-01-01"),
        "timing_class": "totally_made_up",
        "fiscal_period_end": pd.Timestamp("2019-12-31"),
        "actual_eps": 1.0, "estimated_eps": 0.9,
        "actual_revenue": 1e9, "estimated_revenue": 9e8,
        "estimate_timestamp": pd.Timestamp("2019-12-30"),
        "filing_timestamp": pd.Timestamp("2020-01-01"),
    }])
    with pytest.raises(Exception):
        SCHEMAS["events"].validate(bad)


def test_features_schema_requires_pit_columns():
    bad = pd.DataFrame([{"event_id": "E1", "security_id": "S1"}])
    with pytest.raises(Exception):
        SCHEMAS["features"].validate(bad)


def test_event_master_schema_accepts_valid_row():
    good = pd.DataFrame([{
        "event_id": "E1", "security_id": "S1",
        "effective_trading_date": pd.Timestamp("2020-01-14"),
        "entry_session": pd.Timestamp("2020-01-14"),
        "prediction_timestamp": pd.Timestamp("2020-01-14 09:35"),
        "order_submission_cutoff": pd.Timestamp("2020-01-14 09:25"),
        "timing_class": "before_market_open",
        "exclusion_reason": "",
    }])
    SCHEMAS["event_master"].validate(good)
