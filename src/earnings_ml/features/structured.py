"""Structured features with point-in-time invariants.

Every feature carries:
- feature_available_at: timestamp when the value was observable
- source_available_at:   timestamp when the underlying source became public

All feature_available_at values are STRICTLY LESS than the order_submission_cutoff
(effective_trading_date 09:25 ET), which is itself before the entry session open.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def build_surprise_features(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = events[["event_id", "security_id", "ticker", "sector", "industry",
                 "market_cap_at_event", "cap_bucket", "fiscal_year", "fiscal_quarter",
                 "effective_trading_date", "prediction_timestamp", "order_submission_cutoff",
                 "actual_eps", "estimated_eps", "actual_revenue", "estimated_revenue",
                 "estimate_timestamp", "announcement_timestamp"]].copy()

    eps_surprise = df["actual_eps"] - df["estimated_eps"]
    # % surprise: guard near-zero / negative denominators
    eps_surprise_pct = np.where(
        df["estimated_eps"].abs() > 1e-6,
        eps_surprise / df["estimated_eps"].abs(),
        np.nan,
    )
    rev_surprise = df["actual_revenue"] - df["estimated_revenue"]
    rev_surprise_pct = np.where(
        df["estimated_revenue"].abs() > 1e-6,
        rev_surprise / df["estimated_revenue"].abs(),
        np.nan,
    )

    df["eps_surprise"] = eps_surprise
    df["eps_surprise_pct"] = eps_surprise_pct
    df["rev_surprise"] = rev_surprise
    df["rev_surprise_pct"] = rev_surprise_pct

    # SUE: surprise / std of past surprises (per security). std computed from PRIOR events only.
    df = df.sort_values(["security_id", "effective_trading_date"])
    grp = df.groupby("security_id")["eps_surprise_pct"]
    df["surprise_std_hist"] = grp.transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    df["sue"] = df["eps_surprise_pct"] / df["surprise_std_hist"].replace(0, np.nan)
    df["beat"] = (df["eps_surprise"] > 0).astype(int)
    # consecutive beats
    df["beat_streak"] = (
        df.sort_values(["security_id", "effective_trading_date"])
          .groupby("security_id")["beat"]
          .transform(lambda b: b.groupby((b != b.shift()).cumsum()).cumsum() * b)
    )
    df["prev_surprise_pct"] = grp.transform(lambda s: s.shift(1))

    # Surprise features depend on ACTUALS (eps/revenue), which are only public at the
    # announcement — NOT at estimate_timestamp. Stamp availability accordingly.
    df["feature_available_at"] = df["announcement_timestamp"]
    df["source_available_at"] = df["estimate_timestamp"]
    return df


def build_market_context_features(events: pd.DataFrame, prices: pd.DataFrame,
                                   benchmark: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    pt = prices.copy()
    pt["date"] = pd.to_datetime(pt["date"]).dt.tz_localize(None).dt.normalize()
    if "security_id" not in pt.columns:
        pt["security_id"] = pt["ticker"]
    pt = pt.sort_values(["security_id", "date"]).set_index("date")

    rows = []
    windows = cfg.features.get("rolling_windows", {})
    w_short = int(windows.get("momentum_short", 21))
    w_mid = int(windows.get("momentum_mid", 63))
    w_long = int(windows.get("momentum_long", 126))
    w_vol = int(windows.get("volatility", 63))
    w_vs = int(windows.get("volume_surprise", 63))

    for _, e in events.iterrows():
        sid = e["security_id"]
        entry = pd.Timestamp(e["effective_trading_date"]).tz_localize(None).normalize()
        sub = pt.loc[pt["security_id"] == sid].sort_index()
        hist = sub[sub.index < entry]  # STRICTLY BEFORE entry (no leakage)
        if len(hist) < max(w_long, w_vol, w_vs) + 5:
            rows.append({"event_id": e["event_id"]})
            continue
        close = hist["close"]
        rets = close.pct_change()
        mom_short = close.iloc[-1] / close.iloc[-1 - w_short] - 1 if len(close) > w_short else np.nan
        mom_mid = close.iloc[-1] / close.iloc[-1 - w_mid] - 1 if len(close) > w_mid else np.nan
        mom_long = close.iloc[-1] / close.iloc[-1 - w_long] - 1 if len(close) > w_long else np.nan
        roll_max = close.rolling(w_long).max()
        dist_52w_high = close.iloc[-1] / roll_max.iloc[-1] - 1
        rv = rets.iloc[-w_vol:].std()
        downside = rets[rets < 0].iloc[-w_vol:].std()
        vol_surprise = hist["volume"].iloc[-1] / hist["volume"].iloc[-w_vs:].mean() - 1
        st_reversal = rets.iloc[-5:].sum()
        rows.append({
            "event_id": e["event_id"],
            "mom_1m": mom_short,
            "mom_3m": mom_mid,
            "mom_6m": mom_long,
            "dist_52w_high": dist_52w_high,
            "rv_3m": rv,
            "downside_vol_3m": downside,
            "volume_surprise": vol_surprise,
            "short_term_reversal": st_reversal,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Market-context features are observable when the LAST used close is published,
    # i.e. at the close of the session BEFORE entry. We stamp that explicitly so the
    # PIT invariant reflects actual source availability, not a hardcoded cutoff.
    # Map each event to its last-used close date (= previous session vs entry_session).
    from ..calendars import exchange as ex  # noqa: PLC0415
    avail = []
    for _, e in events[["event_id", "entry_session"]].iterrows():
        entry = pd.Timestamp(e["entry_session"]).tz_localize(None).normalize()
        prev = ex.previous_session(entry)
        # close is published at 16:00 ET on `prev`
        ts = pd.Timestamp(prev).tz_localize(cfg.tz) + pd.Timedelta(hours=16, minutes=5)
        avail.append(ts)
    out["feature_available_at"] = out["event_id"].map(dict(zip(events["event_id"], avail, strict=False)))
    return out


def build_cross_sectional(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = events[["event_id", "security_id", "sector", "industry", "market_cap_at_event",
                 "cap_bucket", "effective_trading_date", "order_submission_cutoff"]].copy()
    df["liquidity_bucket"] = pd.qcut(df["market_cap_at_event"].rank(method="first"), 3,
                                      labels=["low", "mid", "high"])
    # Cross-sectional features (sector/mcap) reflect state-as-of the effective date.
    # Availability = effective_trading_date 00:00 ET (start of entry session).
    df["feature_available_at"] = pd.to_datetime(df["effective_trading_date"]).dt.tz_localize(cfg.tz)
    return df


def merge_features(surprise: pd.DataFrame, market_ctx: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    out = surprise
    if not market_ctx.empty:
        out = out.merge(market_ctx, on="event_id", how="left", suffixes=("", "_mc"))
        if "feature_available_at_mc" in out.columns:
            # A merged row is available only when ALL of its components are available:
            # take the LATEST of the availability timestamps (convert to common tz first).
            a = out["feature_available_at"]
            b = out["feature_available_at_mc"]
            a = pd.to_datetime(a, utc=True)
            b = pd.to_datetime(b, utc=True)
            out["feature_available_at"] = pd.concat([a, b], axis=1).max(axis=1).dt.tz_convert("America/New_York")
            out = out.drop(columns=["feature_available_at_mc"])
    if not cross.empty:
        out = out.merge(cross, on="event_id", how="left", suffixes=("", "_cs"))
        if "feature_available_at_cs" in out.columns:
            a = pd.to_datetime(out["feature_available_at"], utc=True)
            b = pd.to_datetime(out["feature_available_at_cs"], utc=True)
            out["feature_available_at"] = pd.concat([a, b], axis=1).max(axis=1).dt.tz_convert("America/New_York")
            out = out.drop(columns=["feature_available_at_cs"])
    # final source_available_at = min across sources
    if "source_available_at" not in out.columns:
        out["source_available_at"] = out["feature_available_at"]
    return out


def build_structured_features(events: pd.DataFrame, prices: pd.DataFrame,
                               benchmark: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    primary = events.loc[events["exclusion_reason"].fillna("").eq("")].copy()
    if primary.empty:
        return pd.DataFrame()
    surprise = build_surprise_features(primary, cfg)
    market_ctx = build_market_context_features(primary, prices, benchmark, cfg)
    cross = build_cross_sectional(primary, cfg)
    feats = merge_features(surprise, market_ctx, cross)
    return feats
