"""Data contracts: pandera schemas for every intermediate table.

These schemas are the source of truth for column names and types. They are checked in
ingestion, event alignment, features, and labels. Violations are hard errors.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check

# ---- Raw provider outputs ----------------------------------------------------

events_schema = DataFrameSchema(
    {
        "event_id": Column(str),
        "security_id": Column(str),
        "ticker": Column(str),
        "announcement_timestamp": Column(pa.DateTime),
        "timing_class": Column(str, Check.isin(["before_market_open", "after_market_close", "intraday", "unknown"])),
        "fiscal_period_end": Column(pa.DateTime),
        "actual_eps": Column(float, nullable=True),
        "estimated_eps": Column(float, nullable=True),
        "actual_revenue": Column(float, nullable=True),
        "estimated_revenue": Column(float, nullable=True),
        "estimate_timestamp": Column(pa.DateTime, nullable=True),
        "filing_timestamp": Column(pa.DateTime, nullable=True),
    },
    strict=False,
    coerce=True,
)

prices_schema = DataFrameSchema(
    {
        "security_id": Column(str, required=False),
        "ticker": Column(str, required=False),
        "date": Column(pa.DateTime),
        "open": Column(float),
        "high": Column(float),
        "low": Column(float),
        "close": Column(float),
        "volume": Column(float),
    },
    strict=False,
    coerce=True,
)

estimates_schema = DataFrameSchema(
    {
        "event_id": Column(str),
        "security_id": Column(str),
        "estimate_timestamp": Column(pa.DateTime),
        "estimated_eps": Column(float, nullable=True),
        "estimated_revenue": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)

filings_schema = DataFrameSchema(
    {
        "filing_accession_number": Column(str),
        "event_id": Column(str),
        "filing_timestamp": Column(pa.DateTime),
        "text": Column(str),
    },
    strict=False,
    coerce=True,
)

# ---- Aligned event master ----------------------------------------------------

event_master_schema = DataFrameSchema(
    {
        "event_id": Column(str),
        "security_id": Column(str),
        "effective_trading_date": Column(pa.DateTime),
        "entry_session": Column(pa.DateTime),
        "prediction_timestamp": Column(pa.DateTime, nullable=True),
        "order_submission_cutoff": Column(pa.DateTime, nullable=True),
        "timing_class": Column(str),
        "exclusion_reason": Column(str, nullable=True),
    },
    strict=False,
    coerce=True,
)

# ---- Features ----------------------------------------------------------------

feature_required = {
    "event_id",
    "security_id",
    "prediction_timestamp",
    "feature_available_at",
    "source_available_at",
}

features_schema = DataFrameSchema(
    {
        "event_id": Column(str),
        "security_id": Column(str),
        "prediction_timestamp": Column(pa.DateTime),
        "feature_available_at": Column(pa.DateTime),
        "source_available_at": Column(pa.DateTime),
    },
    strict=False,
    coerce=True,
)

# ---- Labels ------------------------------------------------------------------

labels_schema = DataFrameSchema(
    {
        "event_id": Column(str),
        "reaction_car": Column(float, nullable=True),
        "reaction_car_market_adj": Column(float, nullable=True),
        "reaction_car_sector_adj": Column(float, nullable=True),
        "drift_car_20": Column(float, nullable=True),
        "drift_car_20_market_adj": Column(float, nullable=True),
        "drift_car_20_sector_adj": Column(float, nullable=True),
        "realized_vol_20": Column(float, nullable=True),
        "realized_vol_20_annualized": Column(float, nullable=True),
    },
    strict=False,
    coerce=True,
)


SCHEMAS = {
    "events": events_schema,
    "prices": prices_schema,
    "estimates": estimates_schema,
    "filings": filings_schema,
    "event_master": event_master_schema,
    "features": features_schema,
    "labels": labels_schema,
}


def validate(name: str, df):
    schema = SCHEMAS.get(name)
    if schema is None:
        raise KeyError(f"no schema named {name!r}")
    return schema.validate(df)
