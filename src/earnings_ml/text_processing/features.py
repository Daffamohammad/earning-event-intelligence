"""Text features (OFF by default — see configs/features.yaml and ADR-002).

Builds:
- dictionary tone (frozen)
- delta-tone vs prior comparable filing (per security)
- TF-IDF features (vocabulary fit INSIDE each training fold by the model layer)

The TF-IDF transformer itself lives in models/text.py and is fit on the training fold
inside the walk-forward harness to prevent future documents from influencing the vocabulary.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from .dictionary import tone_net, tone_scores


def build_text_features(events: pd.DataFrame, filings: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not cfg.features.get("text", {}).get("enabled", False):
        return pd.DataFrame()
    primary = events.loc[events["exclusion_reason"].fillna("").eq("")].copy()
    merged = primary.merge(filings[["event_id", "filing_timestamp", "text"]], on="event_id", how="left")
    merged["filing_timestamp"] = pd.to_datetime(merged["filing_timestamp"])
    merged["order_submission_cutoff"] = pd.to_datetime(merged["order_submission_cutoff"])
    # Keep only filings strictly available before order cutoff
    merged = merged.loc[merged["filing_timestamp"] < merged["order_submission_cutoff"]]
    if merged.empty:
        return merged
    scores = merged["text"].fillna("").apply(tone_scores).apply(pd.Series)
    merged = pd.concat([merged.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    merged["tone_net"] = merged.apply(lambda r: tone_net(r.to_dict()), axis=1)
    # delta tone vs previous filing for the same security
    merged = merged.sort_values(["security_id", "filing_timestamp"])
    merged["tone_net_prev"] = merged.groupby("security_id")["tone_net"].shift(1)
    merged["delta_tone"] = merged["tone_net"] - merged["tone_net_prev"]
    merged["feature_available_at"] = merged["filing_timestamp"]
    merged["source_available_at"] = merged["filing_timestamp"]
    return merged
