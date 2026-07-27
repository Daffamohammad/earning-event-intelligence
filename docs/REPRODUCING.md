# Reproducing the full pipeline

## 1. Install

```bash
cd "/Users/daffa/Earning Event intelligence"
pip install -e ".[dev]"
```

## 2. Run the full synthetic pipeline

```bash
earnings-ml run-all
```

This executes, in order:
1. `inspect`     — print config + provider summary
2. `ingest`      — run MockProvider, write raw parquet
3. `build-events`— build event master with effective dates
4. `validate-events` — golden + PIT invariant checks
5. `build-features`  — structured features (text OFF by default)
6. `validate-features` — strict PIT feature-availability check
7. `build-labels` — reaction CAR, drift CAR, realized vol
8. `train`       — walk-forward expanding window for all models
9. `backtest`    — portfolio backtest with cost sweep
10. `report`     — HTML report from stored artifacts

## 3. Run tests

```bash
pytest -q
```

Key tests:
- `tests/test_leakage.py` — PIT invariants, estimate-before-announcement, future filings
- `tests/test_calendar.py` — effective-date rules on real historical announcements
- `tests/test_walkforward_isolation.py` — no train/test overlap, no shuffle splits
- `tests/test_negative_control.py` — γ=0 → no manufactured alpha
- `tests/test_report_consistency.py` — report numbers match stored metrics

## 4. Switch to the real-data path

```bash
# edit configs/base.yaml -> provider: edgar_yf
# edit configs/features.yaml -> text.enabled: true
earnings-ml run-all
```

## 5. Negative control

```bash
# edit configs/base.yaml -> signal.gamma_base: 0.0
earnings-ml run-all
pytest tests/test_negative_control.py
```

## Reproducibility notes
- Random seed pinned in `configs/base.yaml -> seed: 20240117`.
- Config hash is written to every metrics JSON.
- All intermediate tables are parquet under `artifacts/data/`.
- Every fold's predictions and metrics are stored under `artifacts/predictions/` and `artifacts/metrics/`.
