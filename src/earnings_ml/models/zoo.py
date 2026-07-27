"""Model zoo.

Baselines (always present):
  - UnconditionalMean       : predicts cross-sectional training mean
  - SignOfSurprise          : direction = sign(eps_surprise)
  - SUE decile              : rank-based SUE score (0..1) from training distribution
  - PriorReaction           : predicts the prior-quarter SURPRISE for the same security
                              (name kept for artifact compatibility)
  - SectorRelativeSurprise  : surprise rank within sector (training-fold)
  - EventDayUnconditional   : buy-every-event drift baseline (no selection)

Structured:
  - LinearRegression, LogisticReaction
  - ElasticNet
  - RandomForest
  - LightGBM

All complex models expose `fit(X, y)` and `predict(X)` and never look at the test fold
during construction. Hyperparameters are FIXED (no validation-fold tuning is currently
performed by the harness; the val window is reserved for future use).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler

# Default feature columns used by structured models
NUMERIC_BASE = [
    "eps_surprise", "eps_surprise_pct", "rev_surprise_pct", "sue", "beat", "beat_streak",
    "prev_surprise_pct", "mom_1m", "mom_3m", "mom_6m", "dist_52w_high", "rv_3m",
    "downside_vol_3m", "volume_surprise", "short_term_reversal", "market_cap_at_event",
]


class BaseModel:
    name: str = "base"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> BaseModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class UnconditionalMean(BaseModel):
    name = "unconditional_mean"

    def fit(self, X, y):
        self._mu = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self._mu)


class SignOfSurprise(BaseModel):
    name = "sign_of_surprise"

    def predict(self, X):
        col = "eps_surprise_pct" if "eps_surprise_pct" in X.columns else "eps_surprise"
        return X[col].fillna(0).to_numpy()


class SUEDecile(BaseModel):
    name = "sue_decile"

    def fit(self, X, y):
        col = "eps_surprise_pct" if "eps_surprise_pct" in X.columns else "eps_surprise"
        self._quantiles = np.nanquantile(X[col].to_numpy(), np.linspace(0, 1, 11))
        return self

    def predict(self, X):
        col = "eps_surprise_pct" if "eps_surprise_pct" in X.columns else "eps_surprise"
        v = X[col].to_numpy()
        return np.digitize(np.nan_to_num(v, nan=0.0), self._quantiles[1:-1]) / 10.0


class PriorReaction(BaseModel):
    name = "prior_reaction"

    def fit(self, X, y):
        # store the prior-reaction signal average per security from training data
        df = X.copy()
        df["_y"] = y.to_numpy() if isinstance(y, pd.Series) else y
        df["_prev"] = df.groupby("security_id")["_y"].shift(1)
        self._global = float(df["_prev"].mean()) if "_prev" in df else 0.0
        return self

    def predict(self, X):
        return X.get("prev_surprise_pct", pd.Series(np.zeros(len(X)))).fillna(0).to_numpy()


class SectorRelativeSurprise(BaseModel):
    name = "sector_relative_surprise"

    def fit(self, X, y):
        col = "eps_surprise_pct"
        self._sector_means = X.groupby("sector")[col].mean().to_dict()
        self._global_mean = float(X[col].mean())
        return self

    def predict(self, X):
        col = "eps_surprise_pct"
        adj = X["sector"].map(self._sector_means).fillna(self._global_mean)
        return (X[col] - adj).fillna(0).to_numpy()


class EventDayUnconditional(BaseModel):
    name = "event_day_unconditional"

    def fit(self, X, y):
        self._mu = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X):
        return np.full(len(X), self._mu)


class LinearReg(BaseModel):
    name = "linear"

    def fit(self, X, y):
        feats = self._feats(X)
        self._scaler = StandardScaler().fit(feats)
        self._model = LinearRegression().fit(self._scaler.transform(feats), y)
        return self

    def predict(self, X):
        feats = self._feats(X)
        return self._model.predict(self._scaler.transform(feats))

    def _feats(self, X):
        cols = [c for c in NUMERIC_BASE if c in X.columns]
        return X[cols].fillna(0).to_numpy()


class ElasticNetReg(LinearReg):
    name = "elastic_net"

    def fit(self, X, y):
        feats = self._feats(X)
        self._scaler = StandardScaler().fit(feats)
        self._model = ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42).fit(
            self._scaler.transform(feats), y
        )
        return self


class RandomForestReg(LinearReg):
    name = "random_forest"

    def fit(self, X, y):
        feats = self._feats(X)
        self._model = RandomForestRegressor(
            n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1
        ).fit(feats, y)
        # no scaler needed but kept for symmetry
        self._scaler = StandardScaler().fit(feats)
        return self

    def predict(self, X):
        feats = self._feats(X)
        return self._model.predict(feats)


class LightGBMReg(LinearReg):
    name = "lightgbm"

    def fit(self, X, y):
        import lightgbm as lgb  # noqa: PLC0415
        feats = self._feats(X)
        self._feats_cols = [c for c in NUMERIC_BASE if c in X.columns]
        self._model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbosity=-1
        ).fit(feats, y)
        self._scaler = StandardScaler().fit(feats)
        return self

    def predict(self, X):
        feats = self._feats(X)
        return self._model.predict(feats)


def make_baselines() -> list[BaseModel]:
    return [
        UnconditionalMean(),
        SignOfSurprise(),
        SUEDecile(),
        PriorReaction(),
        SectorRelativeSurprise(),
        EventDayUnconditional(),
    ]


def make_structured_models() -> list[BaseModel]:
    return [LinearReg(), ElasticNetReg(), RandomForestReg(), LightGBMReg()]


def make_all_models() -> list[BaseModel]:
    return make_baselines() + make_structured_models()
