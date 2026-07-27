"""Point-in-time invariant enforcement.

Every feature row MUST satisfy: feature_available_at <= prediction_timestamp.
A STRICTER check (per confirmation agent) requires feature_available_at < order_submission_cutoff.
"""
from __future__ import annotations

import pandas as pd


def check_feature_availability(features: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """Return rows that VIOLATE the PIT invariant. Empty == pass."""
    if "feature_available_at" not in features or "prediction_timestamp" not in features:
        raise KeyError("features must contain feature_available_at and prediction_timestamp")
    if strict and "order_submission_cutoff" in features.columns:
        bad = features.loc[features["feature_available_at"] >= features["order_submission_cutoff"]]
    else:
        bad = features.loc[features["feature_available_at"] > features["prediction_timestamp"]]
    return bad


def check_estimate_pre_announcement(events: pd.DataFrame) -> pd.DataFrame:
    """Return rows where estimate_timestamp is not strictly before announcement_timestamp."""
    bad = events.loc[events["estimate_timestamp"] >= events["announcement_timestamp"]]
    return bad


def check_filing_pre_prediction(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Return rows where filing_timestamp >= prediction_timestamp (future filing leakage)."""
    if "filing_timestamp" not in events:
        return pd.DataFrame()
    merged = features.merge(events[["event_id", "filing_timestamp"]], on="event_id", how="left")
    bad = merged.loc[merged["filing_timestamp"] >= merged["prediction_timestamp"]]
    return bad


def check_feature_label_partial_order(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """For every row, max(feature_available_at) < label_start. label_start = prediction_timestamp."""
    if labels is None or labels.empty:
        return pd.DataFrame()
    merged = features[["event_id", "feature_available_at", "prediction_timestamp"]].merge(
        labels[["event_id"]], on="event_id", how="inner"
    )
    return merged.loc[merged["feature_available_at"] >= merged["prediction_timestamp"]]
