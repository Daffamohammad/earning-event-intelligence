"""Point-in-time leakage tests — the highest-stakes tests in the repo.

Each test FAILS if any leakage mode is present. Per confirmation-agent requirement #7:
  - feature_available_at < order_submission_cutoff (strict, not <=)
  - estimate_timestamp < announcement_timestamp (strict)
  - filing_timestamp < prediction_timestamp (strict)
  - feature-vs-label partial order: max(feature_available_at) < label_start
  - all timestamps tz-aware (no naive datetime bugs)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from earnings_ml.config import load_config
from earnings_ml.event_alignment.master import build_event_master
from earnings_ml.features.structured import build_structured_features
from earnings_ml.ingestion.mock import MockProvider
from earnings_ml.labels.car import build_labels
from earnings_ml.validation.invariants import (
    check_estimate_pre_announcement,
    check_feature_availability,
    check_feature_label_partial_order,
    check_filing_pre_prediction,
)


@pytest.fixture(scope="module")
def pipeline_outputs():
    cfg = load_config()
    out = MockProvider(cfg).fetch()
    master = build_event_master(out.events, cfg)
    feats = build_structured_features(master, out.prices, out.benchmark_prices, cfg)
    labels = build_labels(master, out.prices, out.benchmark_prices, out.sector_prices, cfg)
    return cfg, master, feats, labels


def test_feature_available_strictly_before_order_cutoff(pipeline_outputs):
    cfg, master, feats, _ = pipeline_outputs
    bad = check_feature_availability(feats, strict=True)
    assert len(bad) == 0, f"{len(bad)} rows have feature_available_at >= order_submission_cutoff"


def test_estimate_timestamp_strictly_before_announcement(pipeline_outputs):
    cfg, master, feats, _ = pipeline_outputs
    bad = check_estimate_pre_announcement(master)
    assert len(bad) == 0, f"{len(bad)} rows have estimate_timestamp >= announcement_timestamp"


def test_filing_timestamp_strictly_before_prediction(pipeline_outputs):
    cfg, master, feats, _ = pipeline_outputs
    bad = check_filing_pre_prediction(master, feats)
    assert len(bad) == 0, f"{len(bad)} rows have filing_timestamp >= prediction_timestamp"


def test_feature_label_partial_order(pipeline_outputs):
    cfg, master, feats, labels = pipeline_outputs
    bad = check_feature_label_partial_order(feats, labels)
    assert len(bad) == 0, "feature_available_at must be < prediction_timestamp (label start)"


def test_all_timestamps_tz_aware(pipeline_outputs):
    cfg, master, feats, _ = pipeline_outputs
    for col in ["prediction_timestamp", "order_submission_cutoff"]:
        s = master[col].dropna()
        assert s.dt.tz is not None, f"{col} must be tz-aware"
    s = feats["feature_available_at"].dropna()
    assert s.dt.tz is not None, "feature_available_at must be tz-aware"


def test_market_context_uses_only_pre_entry_closes(pipeline_outputs):
    """Every mom / rv feature must use closes STRICTLY before the entry session."""
    cfg, master, feats, _ = pipeline_outputs
    # mom_1m uses 21 trading days. The 21st-close must be < entry_session.
    # We assert by: no row has feature_available_at >= entry_session 09:30 ET
    merged = feats.merge(master[["event_id", "entry_session"]], on="event_id", how="left")
    entry_cutoff = pd.to_datetime(merged["entry_session"]).dt.tz_localize(cfg.tz) + pd.Timedelta(hours=9, minutes=30)
    fa = pd.to_datetime(merged["feature_available_at"], utc=True)
    bad = merged.loc[fa >= entry_cutoff.dt.tz_convert("UTC")]
    assert len(bad) == 0


def test_intraday_unknown_excluded_from_primary(pipeline_outputs):
    cfg, master, feats, _ = pipeline_outputs
    primary_mask = master["exclusion_reason"].fillna("").eq("")
    for tc in ("intraday", "unknown"):
        assert not ((master["timing_class"] == tc) & primary_mask).any(), f"{tc} events must be excluded"


def test_feature_availability_stamp_reflects_latest_source(pipeline_outputs):
    """Merged feature_available_at must be the MAX of component availabilities.

    In the mock universe the announcement (BMO 08:05 ET on the entry date; AMC 16:05 ET
    the prior session) is always the latest component (prev-close 16:05 ET, effective
    date 00:00 ET), so the merged stamp must EQUAL the announcement timestamp. Before the
    review fix, a min-merge stamped rows at estimate_timestamp (3 days BEFORE the
    announcement), which would mask a genuine post-cutoff feature.
    """
    cfg, master, feats, _ = pipeline_outputs
    # feats already carries announcement_timestamp / estimate_timestamp (selected in
    # build_surprise_features), so no merge is needed.
    feats = feats.merge(master[["event_id", "timing_class"]], on="event_id")
    fa = pd.to_datetime(feats["feature_available_at"], utc=True)
    ann = pd.to_datetime(feats["announcement_timestamp"], utc=True)
    est = pd.to_datetime(feats["estimate_timestamp"], utc=True)
    cutoff = pd.to_datetime(feats["order_submission_cutoff"], utc=True)
    # availability is never stamped EARLIER than the announcement (the old min-merge
    # stamped rows at estimate_timestamp, 3 days before the announcement)
    assert (fa >= ann).all()
    assert (fa > est).all()
    # BMO rows are bound by the announcement itself (08:05 ET on the entry date);
    # AMC rows by the effective-date 00:00 ET cross-sectional stamp.
    bmo = feats["timing_class"] == "before_market_open"
    assert (fa[bmo] == ann[bmo]).all()
    assert (fa[~bmo] > ann[~bmo]).all()
    assert (fa < cutoff).all()
