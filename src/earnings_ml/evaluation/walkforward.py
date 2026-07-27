"""Walk-forward evaluation harness.

Expanding-window design:
  train: [initial_train_start, initial_train_end]
  val:   [initial_val_start,   initial_val_end]
  test:  rolls forward one year at a time, growing the train window (expanding).

For each fold:
  1. Fit preprocessing (scaling) ONLY on train.
  2. Fit model on train.
  3. Predict ONCE on test.
  4. Store fold-level predictions + model config + feature importance.

NOTE: the val window is defined per fold but currently UNUSED — all models run with
fixed hyperparameters. Fold construction still guarantees val < test chronologically so
a future tuner can consume it without redesign.

CRITICAL (per confirmation agent): no random splits, no shuffling, no train_test_split /
ShuffleSplit anywhere in this module. A CI test asserts this.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import Config
from ..models.zoo import BaseModel, make_all_models
from .metrics import fold_metrics, pooled_metrics

_SPLIT_FORBIDDEN = {"train_test_split", "ShuffleSplit", "KFold", "StratifiedKFold", "GroupKFold"}


def assert_no_shuffle_split_in_module(source_text: str) -> None:
    """Hard guard against accidental shuffling.

    Uses AST parsing to find actual function-call / import usages, ignoring mentions
    in docstrings and comments. A CI test asserts this on the module's own source.
    """
    try:
        tree = ast.parse(source_text)
    except SyntaxError as e:
        raise AssertionError(f"could not parse module: {e}") from e
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name in _SPLIT_FORBIDDEN:
                found.add(name)
    if found:
        raise AssertionError(f"forbidden splitter(s) {found} used in evaluation module")


@dataclass
class FoldSpec:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(cfg: Config) -> list[FoldSpec]:
    wf = cfg.walkforward
    train_start = pd.Timestamp(wf["initial_train_start"])
    train_end = pd.Timestamp(wf["initial_train_end"])
    val_start = pd.Timestamp(wf["initial_val_start"])
    val_end = pd.Timestamp(wf["initial_val_end"])
    test_start = pd.Timestamp(wf["first_test_start"])
    last_end = pd.Timestamp(wf["last_test_end"])
    step = pd.DateOffset(years=int(wf.get("fold_step_years", 1)))

    folds = []
    while test_start <= last_end:
        test_end = min(test_start + step - pd.Timedelta(days=1), last_end)
        folds.append(FoldSpec(
            name=f"test_{test_start.year}",
            train_start=train_start, train_end=train_end,
            val_start=val_start, val_end=val_end,
            test_start=test_start, test_end=test_end,
        ))
        # expanding window: grow train and val
        train_end = val_end
        val_start = test_start
        val_end = test_end
        test_start = test_start + step
    return folds


def _filter_by_date(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    s = pd.to_datetime(df[col]).dt.tz_localize(None)
    return df.loc[(s >= start) & (s <= end)]


def run_walkforward(features: pd.DataFrame, labels: pd.DataFrame, cfg: Config,
                    models: list[BaseModel] | None = None,
                    target: str = "reaction_car",
                    predictions_dir: str | Path | None = None) -> dict:
    models = models or make_all_models()
    folds = make_folds(cfg)

    # merge features with labels
    df = features.merge(labels[["event_id", target]], on="event_id", how="inner")
    df["effective_trading_date"] = pd.to_datetime(df["effective_trading_date"]).dt.tz_localize(None)

    all_fold_results = []
    all_predictions = []
    fold_summaries = []

    for fold in folds:
        train = _filter_by_date(df, "effective_trading_date", fold.train_start, fold.train_end)
        test = _filter_by_date(df, "effective_trading_date", fold.test_start, fold.test_end)
        if train.empty or test.empty:
            continue

        X_train = train.drop(columns=[target])
        y_train = train[target]
        X_test = test.drop(columns=[target])
        y_test = test[target]

        for model in models:
            try:
                model.fit(X_train.copy(), y_train.copy())
                preds = model.predict(X_test.copy())
            except Exception as e:  # never crash the harness on a single model
                fold_summaries.append({
                    "fold": fold.name, "model": model.name, "status": f"error: {e!r}",
                    "n_test": int(len(test)),
                })
                continue
            pred_df = pd.DataFrame({
                "event_id": X_test["event_id"].to_numpy(),
                "fold": fold.name,
                "model": model.name,
                "prediction": preds,
                "actual": y_test.to_numpy(),
                "prediction_timestamp": X_test["prediction_timestamp"].to_numpy()
                    if "prediction_timestamp" in X_test.columns else pd.NaT,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
            })
            all_predictions.append(pred_df)
            m = fold_metrics(preds, y_test.to_numpy())
            m.update({"fold": fold.name, "model": model.name, "n_test": int(len(test))})
            all_fold_results.append(m)
            fold_summaries.append({"fold": fold.name, "model": model.name, "status": "ok", "n_test": int(len(test))})

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    fold_results = pd.DataFrame(all_fold_results)

    pooled = pooled_metrics(predictions) if not predictions.empty else {}

    if predictions_dir is not None:
        out = Path(predictions_dir)
        out.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(out / "predictions.parquet", index=False)
        fold_results.to_parquet(out / "fold_metrics.parquet", index=False)

    return {
        "fold_metrics": fold_results,
        "predictions": predictions,
        "pooled_metrics": pooled,
        "folds": [f.name for f in folds],
        "fold_summaries": fold_summaries,
    }


def forbid_shuffle_self_check() -> None:
    """Tests call this; reads THIS file and asserts no forbidden splitters."""
    import inspect  # noqa: PLC0415
    src = inspect.getsource(__import__("earnings_ml.evaluation.walkforward", fromlist=["x"]))
    assert_no_shuffle_split_in_module(src)
