"""Utility helpers: hashing, IO, seeds, provenance."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # noqa: PLC0415
        torch.manual_seed(seed)
    except Exception:
        pass


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def write_parquet(df: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def write_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)


def read_json(path: str | Path) -> Any:
    with open(path) as fh:
        return json.load(fh)


def provenance(cfg_hash: str, files: dict[str, str]) -> dict[str, Any]:
    return {"config_hash": cfg_hash, "files": files}
