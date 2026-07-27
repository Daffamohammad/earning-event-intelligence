"""Ablation and robustness suite.

Required ablations (§17):
- structured only / text only / structured+text
- earnings surprise only / market context only
- remove sector / momentum / valuation
- exclude 2020 / large-cap / small-mid-cap / BMO-only / AMC-only

Required robustness:
- alternative benchmark, alt drift horizons (5/60), alt costs,
  winsorization, training-window length, alt probability threshold,
  sector-neutral vs unconstrained, event clustering, performance decay.

Each ablation/robustness runs the walk-forward harness on a subset of features or events
and stores the pooled metric next to the headline result.
"""
from __future__ import annotations

import pandas as pd

from ..config import Config
from ..evaluation.walkforward import run_walkforward
from ..models.zoo import make_all_models

ABLATION_FEATURE_GROUPS = {
    "earnings_surprise_only": ["eps_surprise", "eps_surprise_pct", "rev_surprise_pct", "sue", "beat", "beat_streak", "prev_surprise_pct"],
    "market_context_only": ["mom_1m", "mom_3m", "mom_6m", "dist_52w_high", "rv_3m", "downside_vol_3m", "volume_surprise", "short_term_reversal"],
}
REMOVE_GROUPS = {
    "no_sector": ["sector", "industry"],
    "no_momentum": ["mom_1m", "mom_3m", "mom_6m"],
    "no_valuation": ["market_cap_at_event", "dist_52w_high"],
}
PIT_KEEP = {"event_id", "security_id", "prediction_timestamp", "feature_available_at",
            "source_available_at", "effective_trading_date", "order_submission_cutoff",
            "ticker", "cap_bucket", "sector", "industry"}


def _subset_features(feats: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    base_cols = [c for c in PIT_KEEP if c in feats.columns]
    cols = base_cols + [c for c in keep if c in feats.columns]
    return feats[cols]


def run_ablations(feats: pd.DataFrame, labels: pd.DataFrame, cfg: Config,
                  target: str = "reaction_car") -> dict:
    results = {}
    models = make_all_models()

    # Feature-group ablations
    for name, keep in ABLATION_FEATURE_GROUPS.items():
        sub = _subset_features(feats, keep)
        r = run_walkforward(sub, labels, cfg, models=models, target=target)
        results[name] = _summarize(r["pooled_metrics"])

    # Feature-removal ablations
    for name, drop in REMOVE_GROUPS.items():
        keep = [c for c in feats.columns if c not in drop and c not in PIT_KEEP]
        sub = _subset_features(feats, keep)
        r = run_walkforward(sub, labels, cfg, models=models, target=target)
        results[name] = _summarize(r["pooled_metrics"])

    # Event-subset ablations
    if "effective_trading_date" in feats.columns:
        ed = pd.to_datetime(feats["effective_trading_date"]).dt.year
        sub = feats[ed != 2020]
        r = run_walkforward(sub, labels, cfg, models=models, target=target)
        results["exclude_2020"] = _summarize(r["pooled_metrics"])

    for bucket in ["large_cap", "mid_cap", "small_cap"]:
        if "cap_bucket" in feats.columns:
            sub = feats[feats["cap_bucket"] == bucket]
            if len(sub) > 100:
                r = run_walkforward(sub, labels, cfg, models=models, target=target)
                results[f"cap_{bucket}"] = _summarize(r["pooled_metrics"])

    return results


def run_robustness(feats: pd.DataFrame, labels: pd.DataFrame, cfg: Config,
                    target: str = "reaction_car") -> dict:
    results = {}
    models = make_all_models()

    # Alternative drift horizons (5, 60)
    for h in [5, 60]:
        t = f"drift_car_{h}"
        if t in labels.columns:
            r = run_walkforward(feats, labels, cfg, models=models, target=t)
            results[f"drift_h{h}"] = _summarize(r["pooled_metrics"])

    # Performance decay over calendar time (per-fold pearson for the best model)
    r = run_walkforward(feats, labels, cfg, models=models, target=target)
    fm = r["fold_metrics"]
    if not fm.empty:
        # pick the model with the highest mean pearson across folds
        agg = fm.groupby("model")["pearson"].mean().sort_values(ascending=False)
        best = agg.index[0]
        per_fold = fm[fm["model"] == best].set_index("fold")["pearson"].to_dict()
        results["decay_per_fold"] = {"model": best, "pearson_by_fold": per_fold}

    return results


def _summarize(pooled: dict) -> dict:
    if not pooled:
        return {}
    out = {}
    for model, m in pooled.items():
        out[model] = {k: m.get(k) for k in ["n_total", "pearson", "pearson_ci_low",
                                             "pearson_ci_high", "top_minus_bottom",
                                             "roc_auc", "ece", "pead_magnitude"]}
    return out
