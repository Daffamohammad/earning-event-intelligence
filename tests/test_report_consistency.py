"""Report-consistency tests: numbers in the HTML report must equal stored metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from earnings_ml.config import load_config
from earnings_ml.utils import read_json


def test_report_exists_after_run_all():
    cfg = load_config()
    report = cfg.artifacts_dir / "report" / "report.html"
    if not report.exists():
        pytest.skip("report not generated yet; run `earnings-ml report`")
    text = report.read_text()
    assert "SYNTHETIC DATA" in text, "report must label synthetic results clearly"
    assert "Config hash" in text


def test_pooled_metrics_match_stored_predictions():
    cfg = load_config()
    train_path = cfg.artifacts_dir / "metrics" / "train_reaction_car.json"
    pred_path = cfg.artifacts_dir / "predictions" / "predictions.parquet"
    if not (train_path.exists() and pred_path.exists()):
        pytest.skip("train artifacts missing")
    import pandas as pd
    stored = read_json(train_path)
    preds = pd.read_parquet(pred_path)
    # n_total in stored metrics must equal len of predictions per model
    for model, m in stored.get("pooled_metrics", {}).items():
        assert m["n_total"] == int((preds["model"] == model).sum()), (
            f"report metric n_total for {model} does not match stored predictions"
        )


def test_backtest_uses_median_cost_as_primary():
    cfg = load_config()
    bt_dir = cfg.artifacts_dir / "backtest"
    if not bt_dir.exists():
        pytest.skip("no backtest artifacts")
    files = list(bt_dir.glob("*.json"))
    assert files, "no backtest JSON files"
    for f in files:
        d = read_json(f)
        # primary_cost_bps must equal the median of the cost sweep
        costs = sorted(int(k) for k in d["cost_sweep"].keys())
        median = costs[len(costs) // 2]
        assert d["primary_cost_bps"] == median, (
            f"{f.name}: primary cost {d['primary_cost_bps']} != median {median}"
        )
