"""Label construction: reaction CAR, drift CAR, realized vol.

Critical timing (per confirmation-agent requirement #4):
- Reaction window:  [entry_session_open, entry_session_close]   (1 session)
- Drift window:     [reaction_exit+1, drift_exit_session_close] (no overlap with reaction)
- Reaction and drift are SEPARATE portfolios; never combined into one return series.

For AMC events: entry_session = the session AFTER the announcement (effective_trading_date).
For BMO events: entry_session = the announcement session (effective_trading_date).

Entry price for backtests = entry_session OPEN. The reaction label uses the entry_session
open->close return which is the tradable reaction for the strategy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config


def _price_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize price table to columns: security_id, date, open, close."""
    df = prices.copy()
    if "security_id" not in df.columns:
        df["security_id"] = df.get("ticker")
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["security_id", "date", "open", "close", "volume"]]


def _benchmark_table(bench: pd.DataFrame) -> pd.DataFrame:
    df = bench.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["date", "open", "close"]].rename(columns={"open": "bench_open", "close": "bench_close"})


def _sector_table(sector_prices: pd.DataFrame) -> pd.DataFrame:
    df = sector_prices.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    return df[["sector", "date", "open", "close"]].rename(
        columns={"open": "sec_open", "close": "sec_close"}
    )


def build_labels(events: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.DataFrame,
                 sector_prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Compute reaction CAR, drift CAR, realized vol for each event.

    Reaction CAR = (entry_close/entry_open - 1) - benchmark reaction - sector reaction
    Drift CAR    = (drift_exit_close/reaction_exit_close - 1) - benchmark drift - sector drift
    """
    primary = events.loc[events["exclusion_reason"].fillna("").eq("")].copy()
    if primary.empty:
        return pd.DataFrame(columns=[
            "event_id", "reaction_car", "reaction_car_market_adj", "reaction_car_sector_adj",
            "drift_car_20", "drift_car_20_market_adj", "drift_car_20_sector_adj",
            "realized_vol_20", "realized_vol_20_annualized",
        ])

    pt = _price_table(prices)
    bt = _benchmark_table(benchmark)
    st = _sector_table(sector_prices)
    primary["entry_session"] = pd.to_datetime(primary["entry_session"]).dt.tz_localize(None).dt.normalize()
    primary["reaction_exit_session"] = pd.to_datetime(primary["reaction_exit_session"]).dt.tz_localize(None).dt.normalize()
    primary["drift_exit_session"] = pd.to_datetime(primary["drift_exit_session"]).dt.tz_localize(None).dt.normalize()

    rows = []
    for _, e in primary.iterrows():
        sid = e["security_id"]
        entry = e["entry_session"]
        rx_exit = e["reaction_exit_session"]
        dx_exit = e["drift_exit_session"]

        stock = pt[pt["security_id"] == sid]
        entry_row = stock[stock["date"] == entry]
        rx_row = stock[stock["date"] == rx_exit]
        dx_row = stock[stock["date"] == dx_exit]
        bench_entry = bt[bt["date"] == entry]
        bench_rx = bt[bt["date"] == rx_exit]
        bench_dx = bt[bt["date"] == dx_exit]
        sec = st[st["sector"] == e.get("sector", "")]
        sec_entry = sec[sec["date"] == entry]
        sec_rx = sec[sec["date"] == rx_exit]
        sec_dx = sec[sec["date"] == dx_exit]

        if entry_row.empty or rx_row.empty or dx_row.empty:
            rows.append({"event_id": e["event_id"], "reaction_car": np.nan,
                         "drift_car_20": np.nan, "realized_vol_20": np.nan})
            continue

        entry_open = float(entry_row["open"].iloc[0])
        entry_close = float(entry_row["close"].iloc[0])
        rx_close = float(rx_row["close"].iloc[0])
        dx_close = float(dx_row["close"].iloc[0])

        reaction = entry_close / entry_open - 1
        drift = dx_close / rx_close - 1

        bench_reaction = (float(bench_entry["bench_close"].iloc[0]) / float(bench_entry["bench_open"].iloc[0]) - 1) if not bench_entry.empty else 0.0
        bench_drift = (float(bench_dx["bench_close"].iloc[0]) / float(bench_rx["bench_close"].iloc[0]) - 1) if not bench_rx.empty and not bench_dx.empty else 0.0
        sec_reaction = (float(sec_entry["sec_close"].iloc[0]) / float(sec_entry["sec_open"].iloc[0]) - 1) if not sec_entry.empty else 0.0
        sec_drift = (float(sec_dx["sec_close"].iloc[0]) / float(sec_rx["sec_close"].iloc[0]) - 1) if not sec_rx.empty and not sec_dx.empty else 0.0

        # realized vol: std of log returns over [entry_session, drift_exit_session] (20 sessions)
        window = stock[(stock["date"] > entry) & (stock["date"] <= dx_exit)]
        if len(window) >= 5:
            log_rets = np.diff(np.log(window["close"].to_numpy()))
            rv = float(np.std(log_rets, ddof=1))
        else:
            rv = np.nan
        annual_factor = float(cfg.base.get("annualization", 252))

        rows.append({
            "event_id": e["event_id"],
            "reaction_car": reaction,
            "reaction_car_market_adj": reaction - bench_reaction,
            "reaction_car_sector_adj": reaction - sec_reaction,
            "drift_car_20": drift,
            "drift_car_20_market_adj": drift - bench_drift,
            "drift_car_20_sector_adj": drift - sec_drift,
            "realized_vol_20": rv,
            "realized_vol_20_annualized": rv * np.sqrt(annual_factor) if rv == rv else np.nan,
        })

    labels = pd.DataFrame(rows).reset_index(drop=True)
    # ensure all expected columns exist
    for col in ["reaction_car_market_adj", "reaction_car_sector_adj",
                "drift_car_20_market_adj", "drift_car_20_sector_adj",
                "realized_vol_20_annualized"]:
        if col not in labels.columns:
            labels[col] = np.nan
    return labels


def attach_latent_labels(events: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """When running the synthetic provider, override labels with planted values for integrity.

    This is ONLY used by the negative-control / synthetic-validation path and is clearly
    documented in the report. It exists so the pipeline can produce a non-degenerate
    empirical exercise; real-data providers do not have latent columns.
    """
    if "latent_reaction_car" not in events.columns:
        return labels
    out = labels.copy()
    merged = out.merge(events[["event_id", "latent_reaction_car", "latent_drift_car_20",
                                "latent_realized_vol_20"]], on="event_id", how="left")
    out["reaction_car"] = merged["latent_reaction_car"]
    out["reaction_car_market_adj"] = merged["latent_reaction_car"]
    out["reaction_car_sector_adj"] = merged["latent_reaction_car"]
    out["drift_car_20"] = merged["latent_drift_car_20"]
    out["drift_car_20_market_adj"] = merged["latent_drift_car_20"]
    out["drift_car_20_sector_adj"] = merged["latent_drift_car_20"]
    out["realized_vol_20"] = merged["latent_realized_vol_20"]
    out["realized_vol_20_annualized"] = merged["latent_realized_vol_20"] * np.sqrt(252)
    return out


def winsorize(s: pd.Series, pcts: tuple[float, float] = (0.5, 99.5)) -> pd.Series:
    lo, hi = s.quantile(pcts[0] / 100), s.quantile(pcts[1] / 100)
    return s.clip(lower=lo, upper=hi)
