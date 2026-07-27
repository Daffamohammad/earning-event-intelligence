"""Walk-forward isolation tests — no train/test overlap, no shuffling."""
from __future__ import annotations

import inspect

import pandas as pd
import pytest

from earnings_ml.config import load_config
from earnings_ml.evaluation.walkforward import (
    FoldSpec,
    assert_no_shuffle_split_in_module,
    make_folds,
)


def test_no_train_test_overlap():
    cfg = load_config()
    folds = make_folds(cfg)
    for f in folds:
        assert f.train_end < f.test_start, f"{f.name}: train overlaps test"
        assert f.val_end < f.test_start, f"{f.name}: val overlaps test"


def test_folds_are_chronological():
    cfg = load_config()
    folds = make_folds(cfg)
    for a, b in zip(folds, folds[1:]):
        assert b.test_start > a.test_start


def test_no_shuffle_split_in_evaluation_module():
    """Hard guard: train_test_split / ShuffleSplit / KFold never imported into the evaluation module."""
    from earnings_ml.evaluation import walkforward as wf
    src = inspect.getsource(wf)
    assert_no_shuffle_split_in_module(src)


def test_folds_cover_2020_through_2025():
    cfg = load_config()
    folds = make_folds(cfg)
    years = [pd.Timestamp(f.test_start).year for f in folds]
    assert years == list(range(2020, 2026))


def test_reaction_horizon_at_least_one_session(cfg):
    """Per Checkpoint B #4: reaction_horizon_sessions >= 1 guarantees disjoint windows."""
    assert int(cfg.labels.get("reaction_horizon_sessions", 1)) >= 1
