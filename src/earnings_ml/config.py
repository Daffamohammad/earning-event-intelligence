"""Configuration loading and hashing."""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class Config:
    base: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    walkforward: dict[str, Any] = field(default_factory=dict)
    portfolio: dict[str, Any] = field(default_factory=dict)

    def get(self, section: str, key: str | None = None, default: Any = None) -> Any:
        data = getattr(self, section, {})
        if key is None:
            return data
        return data.get(key, default)

    @property
    def seed(self) -> int:
        return int(self.base.get("seed", 20240117))

    @property
    def provider(self) -> str:
        return str(self.base.get("provider", "mock"))

    @property
    def artifacts_dir(self) -> Path:
        root = self.base.get("artifacts_dir", "artifacts")
        return Path(root)

    @property
    def tz(self) -> str:
        return str(self.base.get("tz", "America/New_York"))

    @property
    def data_dir(self) -> Path:
        return Path(self.base.get("data_dir", "artifacts/data"))

    @property
    def paths(self) -> dict[str, str]:
        return dict(self.base.get("paths", {}))

    def hash(self) -> str:
        blob = yaml.safe_dump(
            {
                "base": self.base,
                "labels": self.labels,
                "features": self.features,
                "walkforward": self.walkforward,
                "portfolio": self.portfolio,
            },
            sort_keys=True,
            default_flow_style=False,
        ).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(config_dir: str | os.PathLike[str] | Path | None = None) -> Config:
    cdir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    cfg = Config()
    for name, target in [
        ("base", cfg.base),
        ("labels", cfg.labels),
        ("features", cfg.features),
        ("walkforward", cfg.walkforward),
        ("portfolio", cfg.portfolio),
    ]:
        path = cdir / f"{name}.yaml"
        if path.exists():
            with open(path) as fh:
                loaded = yaml.safe_load(fh) or {}
            target.update(loaded)
    return cfg


def ensure_dirs(cfg: Config) -> None:
    base = cfg.artifacts_dir
    for sub in ["data", "metrics", "predictions", "models", "report", "backtest"]:
        (base / sub).mkdir(parents=True, exist_ok=True)
