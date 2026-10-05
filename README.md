# Earnings-Event Intelligence: Post-Announcement Return & Volatility Research

A leakage-tested framework for evaluating post-announcement returns, post-earnings announcement drift (PEAD), and realized volatility.

> **IMPORTANT — scope of empirical claims.**
> This repository ships with a **deterministic synthetic data provider** as the default
> data source because no paid point-in-time data (Compustat / IBES / Refinitiv) or
> security master with delistings is available in this environment. **Every empirical
> number produced by the default pipeline is SYNTHETIC** and is labeled as such in every
> artifact and in the final HTML report.
>
> The framework is a **leakage-tested ML pipeline** validated end-to-end on synthetic
> data, with provider adapters for **SEC EDGAR** (point-in-time-correct filings + tickers)
> and **yfinance** (public prices, flagged for survivorship/revision bias) implemented as
> the real-data path. **No empirical claim about US equities is made by the default
> pipeline.** See `docs/ADR.md` decision ADR-001.

## Quick start

```bash
pip install -e ".[dev]"
earnings-ml run-all            # full reproducible synthetic pipeline
earnings-ml report             # regenerate HTML report from stored artifacts
pytest                         # tests including leakage / negative-control
```

## Layout

```
src/earnings_ml/   package (calendars, ingestion, features, labels, models, ...)
configs/           YAML configuration
data_contracts/    pandera schemas for every intermediate table
tests/             unit, integration, leakage, golden, walk-forward isolation, negative-control
golden/            hand-verified REAL historical announcements (calendar validation)
artifacts/         parquet datasets, fold predictions, model configs, metrics, HTML report
notebooks/         research notebook + colab_quickstart.ipynb (Google-Colab-ready tour)
docs/              ADR, data dictionary, reproducibility instructions
```

See `docs/REPRODUCING.md` for the full pipeline and `docs/ADR.md` for design decisions.
