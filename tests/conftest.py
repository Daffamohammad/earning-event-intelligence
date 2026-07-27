"""Test helpers and fixtures."""
from __future__ import annotations

from pathlib import Path

import pytest

from earnings_ml.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg():
    return load_config(str(ROOT / "configs"))
