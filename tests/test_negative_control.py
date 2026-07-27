"""Negative-control test: when the planted signal gamma = 0, the pipeline must NOT
manufacture alpha above the unconditional baseline.

This is the single most important integrity check for a synthetic-data project
(confirmation-agent requirement #13).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from earnings_ml.config import load_config
from earnings_ml.event_alignment.master import build_event_master
from earnings_ml.features.structured import build_structured_features
from earnings_ml.evaluation.walkforward import run_walkforward
from earnings_ml.ingestion.mock import MockProvider
from earnings_ml.labels.car import attach_latent_labels, build_labels
from earnings_ml.models.zoo import make_baselines, make_structured_models


@pytest.mark.slow
def test_no_alpha_when_signal_removed():
    cfg = load_config()
    cfg.base["signal"]["gamma_base"] = 0.0
    cfg.base["signal"]["drift_multiplier"] = 0.0
    out = MockProvider(cfg).fetch()
    master = build_event_master(out.events, cfg)
    feats = build_structured_features(master, out.prices, out.benchmark_prices, cfg)
    labels_raw = build_labels(master, out.prices, out.benchmark_prices, out.sector_prices, cfg)
    labels = attach_latent_labels(master, labels_raw)  # uses planted (now ~0) labels
    models = make_baselines() + make_structured_models()
    res = run_walkforward(feats, labels, cfg, models=models, target="reaction_car")
    pooled = res["pooled_metrics"]
    # The best model's |pearson| must be small (no signal to learn)
    pearsons = [abs(v.get("pearson", 0) or 0) for v in pooled.values()]
    best = max(pearsons) if pearsons else 0
    # Allow noise but require it to be well below the planted-signal benchmark (~0.5).
    assert best < 0.15, (
        f"negative control produced |pearson|={best:.3f} — pipeline is manufacturing "
        "alpha from noise; this is a critical leakage or label-leak failure."
    )
