"""Portfolio backtest with transaction-cost sensitivity.

Portfolios (per horizon, NEVER mixed):
  1. EW long-only top prediction quantile
  2. EW long bottom quantile (shorting reported SEPARATELY)
  3. Top-minus-bottom
  4. SUE baseline portfolio (SUE-decile top quantile)
  5. Sector-neutral sensitivity

Entry:  next-session OPEN after prediction_timestamp  (strictly after info is tradable)
Exit:   reaction horizon -> reaction_exit_session close ; drift horizon -> drift_exit_session close

Cost sweep: {0, 5, 10, 20, 30, 50} bps round trip. Primary conclusion reported at the
MEDIAN cost (20 bps) per confirmation-agent requirement #5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..labels.car import _price_table


def _entry_exit_returns(events: pd.DataFrame, prices: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """Per-event tradable return from entry open to exit close."""
    pt = _price_table(prices)
    rows = []
    for _, e in events.iterrows():
        sid = e["security_id"]
        sub = pt[pt["security_id"] == sid].set_index("date")
        entry = pd.Timestamp(e["entry_session"]).tz_localize(None).normalize()
        exit_col = "reaction_exit_session" if horizon == "reaction" else "drift_exit_session"
        exit_d = pd.Timestamp(e[exit_col]).tz_localize(None).normalize()
        try:
            entry_open = float(sub.loc[entry, "open"])
            exit_close = float(sub.loc[exit_d, "close"])
        except KeyError:
            rows.append({"event_id": e["event_id"], "gross_return": np.nan,
                         "entry_date": entry, "exit_date": exit_d})
            continue
        rows.append({
            "event_id": e["event_id"], "gross_return": exit_close / entry_open - 1,
            "entry_date": entry, "exit_date": exit_d,
        })
    return pd.DataFrame(rows)


def assign_quantiles(preds: pd.Series, n: int = 5) -> pd.Series:
    return pd.qcut(preds.rank(method="first"), n, labels=False) + 1


def run_backtest(events: pd.DataFrame, predictions: pd.DataFrame, prices: pd.DataFrame,
                 cfg: Config, horizon: str = "drift") -> dict:
    """Run all portfolios for one (model, horizon) combination.

    `predictions` is for ONE model; columns: event_id, prediction, fold, test_start.

    Quantiles are assigned WITHIN each walk-forward fold: all predictions in a test fold
    come from one model trained before the fold, so ranking them against each other uses
    no future information. Pooling across folds would let later predictions alter earlier
    events' quantile membership (lookahead in portfolio formation).
    """
    n_q = int(cfg.portfolio.get("n_quantiles", 5))
    cost_levels = list(cfg.portfolio.get("costs_bps_round_trip", [0, 5, 10, 20, 30, 50]))

    primary = events.loc[events["exclusion_reason"].fillna("").eq("")].copy()
    primary["entry_session"] = pd.to_datetime(primary["entry_session"]).dt.tz_localize(None).dt.normalize()
    primary["drift_exit_session"] = pd.to_datetime(primary["drift_exit_session"]).dt.tz_localize(None).dt.normalize()
    primary["reaction_exit_session"] = pd.to_datetime(primary["reaction_exit_session"]).dt.tz_localize(None).dt.normalize()

    rets = _entry_exit_returns(primary, prices, horizon)
    df = primary.merge(rets, on="event_id", how="inner")
    keep = [c for c in ["event_id", "prediction", "fold", "test_start"] if c in predictions.columns]
    df = df.merge(predictions[keep], on="event_id", how="inner")
    df = df.dropna(subset=["gross_return", "prediction"])

    if df.empty:
        return {"horizon": horizon, "status": "empty"}

    if "fold" in df.columns and df["fold"].notna().any():
        df["quantile"] = df.groupby("fold")["prediction"].transform(lambda p: assign_quantiles(p, n_q))
    else:
        df["quantile"] = assign_quantiles(df["prediction"], n_q)

    results = {}
    by_cost = {}
    for cost_bps in cost_levels:
        cost_dec = cost_bps / 10000.0
        top = df[df["quantile"] == n_q]
        bot = df[df["quantile"] == 1]
        long_ret = (top["gross_return"] - cost_dec).mean() if len(top) else np.nan
        # short leg P&L: gains when the bottom quantile falls; pays its own round-trip cost
        short_ret = (-bot["gross_return"] - cost_dec).mean() if len(bot) else np.nan
        # long-short pays the round-trip cost on BOTH legs
        ls_ret = long_ret + short_ret
        by_cost[cost_bps] = {
            "n_long": int(len(top)),
            "n_short": int(len(bot)),
            "long_only_gross": float(top["gross_return"].mean()) if len(top) else np.nan,
            "long_only_net": float(long_ret),
            "short_only_net": float(short_ret),
            "long_short_net": float(ls_ret),
        }
    results["cost_sweep"] = by_cost
    results["primary_cost_bps"] = int(cfg.portfolio.get("primary_cost_bps", 20))
    results["horizon"] = horizon

    # aggregate stats at primary cost
    primary_cost = results["primary_cost_bps"]
    primary_block = by_cost[primary_cost]
    primary_block.update(_agg_stats(df, primary_cost, n_q))
    results["primary"] = primary_block
    return results


def _agg_stats(df: pd.DataFrame, cost_bps: int, n_q: int) -> dict:
    cost_dec = cost_bps / 10000.0
    top = df[df["quantile"] == n_q].copy()
    bot = df[df["quantile"] == 1].copy()
    top["net"] = top["gross_return"] - cost_dec
    bot["net_short"] = -bot["gross_return"] - cost_dec  # short-leg P&L
    out = {}
    if len(top):
        r = top["net"]
        out["long_annualized"] = float(r.mean() * 4)  # quarterly events -> 4x annual
        out["long_vol"] = float(r.std() * np.sqrt(4))
        out["long_sharpe"] = float(out["long_annualized"] / out["long_vol"]) if out["long_vol"] else np.nan
        out["long_hit_rate"] = float((r > 0).mean())
        out["long_turnover"] = 1.0  # single-period per event
    if len(top) and len(bot):
        ls = top["net"].mean() + bot["net_short"].mean()
        out["long_short_annualized"] = float(ls * 4)
    # by year
    if "test_start" in df.columns and len(top):
        top_y = top.copy()
        top_y["year"] = pd.to_datetime(top_y["test_start"]).dt.year
        out["long_by_year"] = top_y.groupby("year")["net"].mean().to_dict()
    return out
