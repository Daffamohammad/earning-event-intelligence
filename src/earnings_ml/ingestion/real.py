"""Real-price provider: SEC EDGAR (connectivity probe) + yfinance (prices).

This adapter overlays REAL yfinance prices onto the MockProvider's synthetic events,
estimates, and filings, because no free point-in-time earnings/estimate source exists.
It is OFF the default path (provider=mock). Known limitations (docs/ADR.md ADR-003):

- yfinance prices are subject to survivorship and revision bias; no delisting security
  master is available.
- Events/estimates/filings are SYNTHETIC, so `is_synthetic=True` is kept and every
  derived number stays labeled SYNTHETIC (ADR-001). This path is a smoke check for the
  price plumbing, not an empirical anchor.

The implementation is intentionally defensive: network calls are wrapped in retries and
failures fall back to the mock tables.
"""
from __future__ import annotations

import json
import time

import pandas as pd

from ..logging import get_logger
from .base import DataProvider, ProviderOutput
from .mock import MockProvider

log = get_logger(__name__)


class EdgarYFinanceProvider(DataProvider):
    name = "edgar_yf"
    # Events/estimates/filings remain synthetic (only prices are real), so the output is
    # labeled synthetic to keep the SYNTHETIC banner on every derived artifact.
    is_synthetic = True

    def fetch(self) -> ProviderOutput:
        log.warning("edgar_yf provider: real price path; survivorship/revision bias applies")
        # Events / estimates / filings come from the MockProvider (no free PIT earnings
        # source exists), so the output remains SYNTHETIC and MUST stay labeled as such:
        # is_synthetic=True keeps the SYNTHETIC banner in the report (ADR-001).
        mock = MockProvider(self.cfg).fetch()

        # The EDGAR tickers probe is a connectivity smoke check only. Its result is NOT
        # used as the security master: mock security_ids/tickers are synthetic and do not
        # map to real CIKs, and the EDGAR frame lacks the required master columns.
        try:
            edgar = self._edgar_tickers()
            log.info("EDGAR tickers probe ok (%d rows); keeping mock master for coherence", len(edgar))
        except Exception as e:  # network may be unavailable in sandboxed runs
            log.error("EDGAR tickers fetch failed (%s); using mock master", e)
        master = mock.security_master

        try:
            prices, bench, sector = self._yfinance_prices(mock)
        except Exception as e:
            log.error("yfinance fetch failed (%s); using mock prices", e)
            prices, bench, sector = mock.prices, mock.benchmark_prices, mock.sector_prices

        return ProviderOutput(
            events=mock.events, estimates=mock.estimates, prices=prices, filings=mock.filings,
            security_master=master, benchmark_prices=bench, sector_prices=sector,
            provider=self.name, is_synthetic=True,
        )

    # ---- SEC EDGAR --------------------------------------------------------

    def _edgar_tickers(self) -> pd.DataFrame:
        import urllib.request  # noqa: PLC0415
        url = "https://www.sec.gov/files/company_tickers.json"
        req = urllib.request.Request(url, headers={"User-Agent": "research earnings-ml/0.1"})
        for _attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read().decode())
                break
            except Exception:
                time.sleep(1.0)
        else:
            raise RuntimeError("EDGAR tickers unavailable")
        rows = [{"cik": v["cik_str"], "ticker": v["ticker"], "company_name": v["title"]} for v in data.values()]
        return pd.DataFrame(rows)

    # ---- yfinance ---------------------------------------------------------

    def _yfinance_prices(self, mock: ProviderOutput):
        import yfinance as yf  # noqa: PLC0415
        tickers = sorted(mock.security_master["ticker"].unique().tolist())[:100]
        bench_ticker = self.cfg.labels.get("benchmark_ticker", "SPY")
        sector_etfs = list(self.cfg.labels.get("sector_etfs", {}).values()) or ["XLK"]
        start = self.cfg.base["universe"]["start_date"]
        end = self.cfg.base["universe"]["end_date"]

        def dl(tk: str) -> pd.DataFrame:
            df = yf.download(tk, start=start, end=end, progress=False, auto_adjust=False)
            if df is None or df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index().rename(columns={"Date": "date"})
            df["ticker"] = tk
            return df

        price_frames = [dl(t) for t in tickers]
        prices = pd.concat([f for f in price_frames if not f.empty], ignore_index=True)
        bench = dl(bench_ticker)
        bench["ticker"] = bench_ticker
        sector_frames = []
        for etf in sector_etfs:
            f = dl(etf)
            if not f.empty:
                f["ticker"] = etf
                sector_frames.append(f)
        sector = pd.concat(sector_frames, ignore_index=True) if sector_frames else mock.sector_prices
        return prices, bench, sector
