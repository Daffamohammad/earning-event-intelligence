"""Golden dataset validation against 60 REAL historical announcements.

The golden set is hand-sourced from public press releases / IR pages for well-known names.
It is used ONLY to validate calendar/effective-date logic — NOT to validate the synthetic
provider (which would be circular). Each row has a known correct effective_trading_date.

See golden/golden_events.csv for the data.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..calendars import effective_date as ed
from ..config import Config


def load_golden(path: str | Path | None = None, cfg: Config | None = None) -> pd.DataFrame:
    p = Path(path) if path else Path(cfg.base["paths"]["golden_csv"]) if cfg else Path("golden/golden_events.csv")
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    # The CSV stores wall-clock US Eastern times (the actual press-release times).
    df["announcement_timestamp"] = pd.to_datetime(df["announcement_timestamp"]).dt.tz_localize(tz_str(cfg))
    df["expected_effective_date"] = pd.to_datetime(df["expected_effective_date"]).dt.tz_localize(None)
    return df


def tz_str(cfg: Config | None, default: str = "America/New_York") -> str:
    return cfg.tz if cfg is not None else default


def evaluate_golden(cfg: Config) -> dict:
    g = load_golden(cfg=cfg)
    if g.empty:
        return {"status": "missing", "n": 0}
    tz = cfg.tz
    effs = []
    for _, r in g.iterrows():
        eff = ed.effective_trading_date(
            pd.Timestamp(r["announcement_timestamp"]).tz_convert(tz),
            r["timing_class"], tz=tz, exchange="XNYS",
        )
        effs.append(pd.Timestamp(eff).normalize())
    g["computed_effective_date"] = pd.to_datetime(effs)
    g["eff_date_correct"] = (g["computed_effective_date"] == g["expected_effective_date"].dt.normalize())
    g["timing_correct"] = g["timing_class"].isin(["before_market_open", "after_market_close", "intraday", "unknown"])

    n = len(g)
    report = {
        "status": "ok",
        "n": n,
        "effective_date_accuracy": float(g["eff_date_correct"].mean()),
        "timing_class_valid_rate": float(g["timing_correct"].mean()),
        "duplicate_rate": float(g.duplicated(subset=["ticker", "announcement_timestamp"]).mean()),
        "failures": g.loc[~g["eff_date_correct"], ["ticker", "announcement_timestamp", "timing_class",
                                                    "expected_effective_date", "computed_effective_date"]].to_dict("records"),
    }
    return report
