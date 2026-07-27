"""In-memory point-in-time security master.

Each row carries a `valid_at` date so historical sector / market-cap classifications reflect
the event date, never the current state of the world. The EDGAR adapter can populate this
from `company_tickers.json` (CIK->ticker) historical snapshots.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config

COLUMNS = [
    "security_id",
    "ticker",
    "cik",
    "company_name",
    "sector",
    "industry",
    "market_cap_usd",
    "cap_bucket",
    "valid_at",
    "listing_date",
    "delisting_date",
]


def cap_bucket(mcap: float) -> str:
    if mcap >= 10_000_000_000:
        return "large_cap"
    if mcap >= 2_000_000_000:
        return "mid_cap"
    return "small_cap"


def as_of(master: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    d = pd.Timestamp(date).normalize()
    return master[(master.valid_at <= d) & ((master.delisting_date.isna()) | (master.delisting_date > d))].copy()


def validate(master: pd.DataFrame) -> None:
    missing = [c for c in COLUMNS if c not in master.columns]
    if missing:
        raise ValueError(f"security master missing columns: {missing}")
    if master.duplicated(subset=["security_id", "valid_at"]).any():
        raise ValueError("security master has duplicate (security_id, valid_at)")


def synthetic_security_master(cfg: Config, stocks: pd.DataFrame) -> pd.DataFrame:
    """Build from a synthetic stocks table (security_id, ticker, sector, mcap, listing_date)."""
    rows = []
    for _, r in stocks.iterrows():
        rows.append({
            "security_id": r.security_id,
            "ticker": r.ticker,
            "cik": r.cik,
            "company_name": r.company_name,
            "sector": r.sector,
            "industry": r.industry,
            "market_cap_usd": r.mcap,
            "cap_bucket": cap_bucket(r.mcap),
            "valid_at": pd.Timestamp(r.listing_date),
            "listing_date": pd.Timestamp(r.listing_date),
            "delisting_date": pd.NaT,
        })
    out = pd.DataFrame(rows, columns=COLUMNS)
    validate(out)
    return out
