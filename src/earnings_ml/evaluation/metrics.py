"""Prediction metrics: classification, regression, ranking, calibration, CIs.

Includes a PEAD-magnitude proxy: mean(actual) / std(actual) of the target per model group
(a t-stat-like property of the actuals; it does NOT use the predictions). Note the Brier
and ECE scores squash raw return predictions through a logistic and are therefore only a
rough calibration proxy, not a fitted calibration model. Block-bootstrap CIs resample
pooled prediction rows in blocks (simplified two-way clustering).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


def _safe(x):
    return np.asarray(x, dtype=float)


def fold_metrics(preds: np.ndarray, actual: np.ndarray) -> dict:
    preds = _safe(preds)
    actual = _safe(actual)
    mask = np.isfinite(preds) & np.isfinite(actual)
    if mask.sum() < 5:
        return {"n": int(mask.sum())}
    p, a = preds[mask], actual[mask]

    out: dict = {"n": int(mask.sum())}
    out["mae"] = float(mean_absolute_error(a, p))
    out["rmse"] = float(np.sqrt(np.mean((p - a) ** 2)))
    if np.std(p) > 0 and np.std(a) > 0:
        out["pearson"] = float(np.corrcoef(p, a)[0, 1])
        # Spearman
        out["spearman"] = float(pd.Series(p).corr(pd.Series(a), method="spearman"))
    else:
        out["pearson"] = float("nan")
        out["spearman"] = float("nan")

    # top-bottom spread (in cross-section)
    if len(p) >= 10:
        q = np.quantile(p, [0.2, 0.8])
        top = a[p >= q[1]]
        bot = a[p <= q[0]]
        out["top_minus_bottom"] = float(top.mean() - bot.mean())

    # binary classification: positive abnormal return
    y = (a > 0).astype(int)
    if len(np.unique(y)) == 2 and np.std(p) > 0:
        try:
            out["roc_auc"] = float(roc_auc_score(y, p))
        except ValueError:
            out["roc_auc"] = float("nan")
        # Brier
        prob = 1 / (1 + np.exp(-p))  # squash predictions to [0,1] via logistic
        out["brier"] = float(brier_score_loss(y, prob))
        # ECE
        out["ece"] = float(expected_calibration_error(y, prob))
    else:
        out["roc_auc"] = float("nan")
        out["brier"] = float("nan")
        out["ece"] = float("nan")
    return out


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return float(ece)


def pooled_metrics(predictions: pd.DataFrame) -> dict:
    if predictions.empty:
        return {}
    out = {}
    for model, g in predictions.groupby("model"):
        m = fold_metrics(g["prediction"].to_numpy(), g["actual"].to_numpy())
        m["n_total"] = int(len(g))
        # PEAD magnitude proxy: mean(actual) / std(actual) of reaction CAR per fold
        m["pead_magnitude"] = float(g["actual"].mean() / (g["actual"].std() + 1e-9))
        # CI via block bootstrap
        ci = block_bootstrap_ci(g["prediction"].to_numpy(), g["actual"].to_numpy(), "pearson")
        m["pearson_ci_low"], m["pearson_ci_high"] = ci
        out[model] = m
    return out


def block_bootstrap_ci(preds: np.ndarray, actual: np.ndarray, metric: str = "pearson",
                       n_boot: int = 200, seed: int = 42) -> tuple[float, float]:
    """Two-way block bootstrap: resample events and years.

    Simplification: since pooled preds mix folds (years), we resample rows in blocks of 50
    to mimic clustered dependence. Returns (2.5%, 97.5%) for the chosen metric.
    """
    rng = np.random.default_rng(seed)
    p, a = _safe(preds), _safe(actual)
    mask = np.isfinite(p) & np.isfinite(a)
    p, a = p[mask], a[mask]
    n = len(p)
    if n < 20:
        return (float("nan"), float("nan"))
    stats = []
    block = max(5, n // 20)
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=n // block + 1)
        offsets = np.arange(block)
        idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        if metric == "pearson" and np.std(p[idx]) > 0 and np.std(a[idx]) > 0:
            stats.append(np.corrcoef(p[idx], a[idx])[0, 1])
        elif metric == "top_minus_bottom" and len(idx) >= 10:
            q = np.quantile(p[idx], [0.2, 0.8])
            top = a[idx][p[idx] >= q[1]]
            bot = a[idx][p[idx] <= q[0]]
            if len(top) and len(bot):
                stats.append(top.mean() - bot.mean())
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))
