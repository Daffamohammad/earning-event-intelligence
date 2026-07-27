"""Build the canonical event-master table from raw provider output.

Steps:
1. Assign effective_trading_date via calendar rules.
2. Set exclusion_reason for intraday/unknown timing.
3. Deduplicate (security_id, fiscal_period_end) keeping the earliest announcement.
4. Stamp prediction_timestamp and order_submission_cutoff.
5. Validate against the event data contract.
"""
from __future__ import annotations

import pandas as pd

from ..calendars import effective_date as ed
from ..calendars import exchange as ex
from ..config import Config


def build_event_master(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = events.copy()
    if df.empty:
        return df

    effs, excl = [], []
    for _, r in df.iterrows():
        eff = ed.effective_trading_date(r["announcement_timestamp"], r["timing_class"], tz=cfg.tz)
        effs.append(pd.Timestamp(eff).normalize())
        is_excluded, reason = ed.should_exclude(r["timing_class"])
        excl.append(reason if is_excluded else "")
    df["effective_trading_date"] = effs
    df["exclusion_reason"] = df.get("exclusion_reason", "").fillna("").astype(str) + pd.Series(excl, index=df.index)
    df.loc[df["exclusion_reason"].str.len() > 0, "exclusion_reason"] = df.loc[df["exclusion_reason"].str.len() > 0, "exclusion_reason"].str.strip()

    # prediction_timestamp / order_submission_cutoff on the effective date
    df["prediction_timestamp"] = [ed.prediction_timestamp(d, cfg) for d in df["effective_trading_date"]]
    df["order_submission_cutoff"] = [ed.order_submission_cutoff(d, cfg) for d in df["effective_trading_date"]]

    # dedupe: keep earliest announcement per (security_id, fiscal_period_end)
    df = df.sort_values("announcement_timestamp").drop_duplicates(
        subset=["security_id", "fiscal_period_end"], keep="first"
    ).reset_index(drop=True)

    # attach entry session (the effective_trading_date) and exit sessions
    reaction_h = int(cfg.labels.get("reaction_horizon_sessions", 1))
    drift_h = int(cfg.labels.get("drift_horizon_sessions", 20))
    df["entry_session"] = df["effective_trading_date"]
    df["reaction_exit_session"] = [ex.shift_sessions(d, reaction_h, "XNYS") for d in df["effective_trading_date"]]
    df["drift_exit_session"] = [ex.shift_sessions(d, drift_h, "XNYS") for d in df["effective_trading_date"]]

    # data quality status
    df.loc[df["exclusion_reason"].str.len() > 0, "data_quality_status"] = "excluded"
    df.loc[df["announcement_timestamp"].isna(), "data_quality_status"] = "missing_announcement_ts"

    return df


def primary_subset(events: pd.DataFrame) -> pd.DataFrame:
    """Subset of events eligible for the primary analysis (no exclusion)."""
    mask = events["exclusion_reason"].fillna("").eq("")
    return events.loc[mask].copy()


def validate_effective_dates(events: pd.DataFrame) -> pd.DataFrame:
    """Return a small report on effective-date sanity."""
    out = events[["event_id", "security_id", "announcement_timestamp", "timing_class", "effective_trading_date"]].copy()

    def _naive(ts: object) -> pd.Timestamp:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            t = t.tz_convert("UTC").tz_localize(None)
        return t.normalize()

    out["is_session"] = out["effective_trading_date"].apply(lambda d: ex.is_session(pd.Timestamp(d)))
    out["amc_next_session_ok"] = out.apply(
        lambda r: _naive(r["effective_trading_date"]) > _naive(r["announcement_timestamp"])
        if r["timing_class"] == "after_market_close" else True, axis=1)
    out["bmo_same_session_ok"] = out.apply(
        lambda r: _naive(r["effective_trading_date"]) == _naive(r["announcement_timestamp"])
        if r["timing_class"] == "before_market_open" else True, axis=1)
    out["intraday_unknown_excluded_ok"] = out.apply(
        lambda r: bool(events.loc[events["event_id"] == r["event_id"], "exclusion_reason"].iloc[0])
        if r["timing_class"] in ("intraday", "unknown") else True, axis=1)
    return out
