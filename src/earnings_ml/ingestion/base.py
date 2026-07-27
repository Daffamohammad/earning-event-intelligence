"""Provider adapter interface.

Every external data source is wrapped behind `DataProvider`. The default provider is the
deterministic `MockProvider`; `EdgarFilingsProvider` and `YFinanceProvider` are real-data
adapters that may require network access.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from ..config import Config


@dataclass
class ProviderOutput:
    events: pd.DataFrame
    estimates: pd.DataFrame
    prices: pd.DataFrame
    filings: pd.DataFrame
    security_master: pd.DataFrame
    benchmark_prices: pd.DataFrame
    sector_prices: pd.DataFrame
    provider: str
    is_synthetic: bool


class DataProvider(ABC):
    name: str = "abstract"
    is_synthetic: bool = False

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    @abstractmethod
    def fetch(self) -> ProviderOutput: ...


def get_provider(cfg: Config) -> DataProvider:
    name = cfg.provider
    if name == "mock":
        from .mock import MockProvider  # noqa: PLC0415
        return MockProvider(cfg)
    if name == "edgar_yf":
        from .real import EdgarYFinanceProvider  # noqa: PLC0415
        return EdgarYFinanceProvider(cfg)
    raise ValueError(f"unknown provider: {name!r}")
