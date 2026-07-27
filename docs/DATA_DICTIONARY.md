# Data dictionary

## event_master (artifacts/data/event_master.parquet)
| column | type | description |
|---|---|---|
| event_id | str | canonical event identifier |
| security_id | str | permanent security identifier |
| ticker | str | ticker symbol (current) |
| ticker_at_event | str | ticker symbol at event date |
| company_name | str |  |
| cik | int | SEC CIK |
| sector | str | GICS sector |
| industry | str | GICS industry |
| market_cap_at_event | float | USD market cap as of event date |
| cap_bucket | str | small_cap / mid_cap / large_cap |
| fiscal_year | int |  |
| fiscal_quarter | int | 1..4 |
| fiscal_period_end | datetime | end of fiscal period |
| announcement_timestamp | datetime | tz-aware announcement time |
| timing_class | str | before_market_open / after_market_close / intraday / unknown |
| effective_trading_date | datetime | first session at which info is tradable |
| entry_session | datetime | session used for entry (== effective_trading_date) |
| reaction_exit_session | datetime | session at reaction horizon (default +1) |
| drift_exit_session | datetime | session at drift horizon (default +20) |
| prediction_timestamp | datetime | effective_trading_date 09:35 ET |
| order_submission_cutoff | datetime | effective_trading_date 09:25 ET |
| exclusion_reason | str | empty = primary; otherwise the reason |
| actual_eps | float |  |
| estimated_eps | float | analyst consensus |
| estimate_timestamp | datetime | strictly before announcement_timestamp |
| filing_timestamp | datetime | strictly before prediction_timestamp |
| latent_reaction_car | float | SYNTHETIC ONLY: planted reaction CAR |
| latent_drift_car_20 | float | SYNTHETIC ONLY: planted 20d drift CAR |
| latent_realized_vol_20 | float | SYNTHETIC ONLY: planted realized vol |

## features (artifacts/data/features.parquet)
| column | type | description |
|---|---|---|
| event_id | str |  |
| prediction_timestamp | datetime |  |
| feature_available_at | datetime | STRICTLY < order_submission_cutoff |
| source_available_at | datetime |  |
| eps_surprise, eps_surprise_pct | float | actual - estimated |
| rev_surprise_pct | float |  |
| sue | float | standardized unexpected earnings |
| beat_streak | int | consecutive beats up to this event |
| prev_surprise_pct | float | previous-quarter % surprise for same security |
| mom_1m / mom_3m / mom_6m | float | momentum using pre-entry closes |
| dist_52w_high | float |  |
| rv_3m | float | realized vol (3m, pre-entry) |
| downside_vol_3m | float |  |
| volume_surprise | float |  |
| short_term_reversal | float | 5-day pre-entry return |
| liquidity_bucket | str | low / mid / high |

## labels (artifacts/data/labels.parquet)
| column | description |
|---|---|
| reaction_car | entry_open → entry_close return |
| reaction_car_market_adj | minus SPY reaction |
| reaction_car_sector_adj | minus sector-ETF reaction |
| drift_car_20 | reaction_exit_close → drift_exit_close (20 sessions) |
| drift_car_20_market_adj |  |
| drift_car_20_sector_adj |  |
| realized_vol_20 | std of log returns over [entry, drift_exit] |
| realized_vol_20_annualized | × √252 |

## predictions (artifacts/predictions/predictions.parquet)
| column | description |
|---|---|
| event_id |  |
| fold | e.g. test_2020 |
| model | model name |
| prediction | float |
| actual | float |
| prediction_timestamp |  |
| test_start / test_end | fold window |

## backtest (artifacts/backtest/<model>_<horizon>.json)
See `backtest/portfolio.py`.
