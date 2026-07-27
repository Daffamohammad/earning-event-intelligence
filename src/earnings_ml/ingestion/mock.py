"""Deterministic synthetic data provider.

Generates a fully point-in-time-correct synthetic universe with a PLANTED signal so the
entire pipeline (event alignment, leakage tests, walk-forward, backtest, ablations) can be
exercised. The planted signal is:

    reaction_CAR ~ gamma(t, segment) * normalize(surprise) + noise

with gamma decaying over calendar time and amplified for small-cap / high-vol regimes.
Set `signal.gamma_base = 0` in config for the negative-control variant.

CRITICAL: every value carries timestamps that respect point-in-time semantics. The provider
NEVER fabricates "future" fields; downstream leakage tests validate this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calendars import exchange as ex
from ..utils import seed_everything
from .base import DataProvider, ProviderOutput

SECTORS = [
    "Information Technology", "Health Care", "Financials", "Consumer Discretionary",
    "Communication Services", "Industrials", "Consumer Staples", "Energy",
    "Utilities", "Materials", "Real Estate",
]
INDUSTRIES = {
    "Information Technology": "Software", "Health Care": "Pharma",
    "Financials": "Banks", "Consumer Discretionary": "Retail",
    "Communication Services": "Media", "Industrials": "Machinery",
    "Consumer Staples": "Food", "Energy": "Oil & Gas",
    "Utilities": "Electric", "Materials": "Chemicals", "Real Estate": "REIT",
}

NAME_PREFIXES = ["Apex", "Vertex", "Nimbus", "Quanta", "Helios", "Orion", "Cobalt",
                 "Atlas", "Solace", "Pioneer", "Meridian", "Crest", "Lumen", "Argent"]


class MockProvider(DataProvider):
    name = "mock"
    is_synthetic = True

    def fetch(self) -> ProviderOutput:
        seed_everything(self.cfg.seed)
        params = self.cfg.base.get("signal", {})
        gamma_base = float(params.get("gamma_base", 0.25))
        decay = float(params.get("gamma_decay_per_year", 0.02))
        drift_mult = float(params.get("drift_multiplier", 0.6))
        seg_mult = params.get("segment_multipliers", {})
        noise_sd = float(params.get("noise_sd", 1.0))

        stocks = self._make_stocks()
        sessions_idx = ex.sessions(self.cfg)
        prices = self._make_prices(stocks, sessions_idx)
        events, estimates = self._make_events(stocks, sessions_idx,
                                              gamma_base, decay, drift_mult, seg_mult, noise_sd)
        filings = self._make_filings(events)
        master = self._security_master(stocks)
        bench = self._make_benchmark(sessions_idx, "SPY")
        sector_prices = self._make_sector_prices(sessions_idx)
        return ProviderOutput(
            events=events, estimates=estimates, prices=prices, filings=filings,
            security_master=master, benchmark_prices=bench, sector_prices=sector_prices,
            provider=self.name, is_synthetic=True,
        )

    # ----- generators ------------------------------------------------------

    def _make_stocks(self) -> pd.DataFrame:
        n = int(self.cfg.base.get("universe", {}).get("n_stocks", 100))
        rng = np.random.default_rng(self.cfg.seed)
        tickers = []
        used = set()
        for i in range(n):
            pre = NAME_PREFIXES[i % len(NAME_PREFIXES)]
            suf = str(10 + i)
            tk = f"{pre[:3].upper()}{suf}"
            while tk in used:
                suf += "X"
                tk = f"{pre[:3].upper()}{suf}"
            used.add(tk)
            tickers.append(tk)
        sectors = rng.choice(SECTORS, size=n)
        # market cap log-uniform 0.3bn to 400bn
        mcaps = (10**rng.uniform(np.log10(3e8), np.log10(4e11), size=n)).astype(float)
        rows = []
        for i in range(n):
            rows.append({
                "security_id": f"S{i:04d}",
                "ticker": tickers[i],
                "cik": int(1_000_000 + i),
                "company_name": f"{NAME_PREFIXES[i % len(NAME_PREFIXES)]} Holdings Inc.",
                "sector": sectors[i],
                "industry": INDUSTRIES[sectors[i]],
                "mcap": float(mcaps[i]),
                "listing_date": pd.Timestamp("2010-01-04"),
                "beta": float(rng.normal(1.0, 0.3)),
                "annual_vol": float(np.exp(rng.normal(np.log(0.30), 0.3))),
                "drift": float(rng.normal(0.06, 0.04)),  # annual expected return
            })
        return pd.DataFrame(rows)

    def _make_prices(self, stocks: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed + 1)
        dt = 1.0 / 252.0
        rows: list[dict] = []
        for _, s in stocks.iterrows():
            mu = float(s.drift)
            vol = float(s.annual_vol)
            price = float(max(5.0, rng.normal(50, 20)))
            base_path = rng.normal(mu * dt, vol * np.sqrt(dt), size=len(sessions))
            # market factor common across names
            for j, sess in enumerate(sessions):
                ret = base_path[j]
                rows.append({
                    "security_id": s.security_id,
                    "date": pd.Timestamp(sess).normalize(),
                    "open": price * (1 + ret * 0.4),
                    "high": price * (1 + max(ret, 0) + abs(rng.normal()) * vol * np.sqrt(dt) * 0.5),
                    "low": price * (1 + min(ret, 0) - abs(rng.normal()) * vol * np.sqrt(dt) * 0.5),
                    "close": price * (1 + ret),
                    "volume": int(max(1e5, rng.lognormal(14, 1))),
                    "unadjusted_close": price * (1 + ret),
                    "adjusted_close": price * (1 + ret),
                    "dividend": 0.0,
                    "split_factor": 1.0,
                })
                price = price * (1 + ret)
        df = pd.DataFrame(rows)
        return df

    def _make_events(self, stocks, sessions, gamma_base, decay, drift_mult, seg_mult, noise_sd) -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(self.cfg.seed + 2)
        start = pd.Timestamp(self.cfg.base.get("universe", {}).get("start_date", "2015-01-01"))
        end = pd.Timestamp(self.cfg.base.get("universe", {}).get("end_date", "2025-12-31"))
        sessions = sessions[(sessions >= start) & (sessions <= end)]
        # earnings: ~once per quarter per stock, on a stable fiscal calendar
        event_rows = []
        est_rows = []
        for _, s in stocks.iterrows():
            cap_b = "large_cap" if s.mcap >= 1e10 else ("mid_cap" if s.mcap >= 2e9 else "small_cap")
            # choose quarterly anchor months
            anchor = rng.choice([1, 2, 3])  # fiscal quarter end offset
            for year in range(start.year, end.year + 1):
                for q in range(4):
                    fq_end_month = ((anchor + q * 3 - 1) % 12) + 1
                    fq_end = pd.Timestamp(year=year, month=fq_end_month, day=1) + pd.offsets.MonthEnd(0)
                    ann_date = fq_end + pd.Timedelta(days=int(rng.integers(20, 50)))
                    # snap to a real session
                    if not ex.is_session(ann_date):
                        ann_date = ex.next_session(ann_date)
                    if ann_date > end or ann_date < start:
                        continue
                    # timing class
                    r = rng.random()
                    if r < 0.4:
                        timing = "before_market_open"
                        ann_ts = pd.Timestamp(ann_date).tz_localize(self.cfg.tz) + pd.Timedelta(hours=8, minutes=5)
                    elif r < 0.85:
                        timing = "after_market_close"
                        ann_ts = pd.Timestamp(ann_date).tz_localize(self.cfg.tz) + pd.Timedelta(hours=16, minutes=5)
                    elif r < 0.95:
                        timing = "intraday"
                        ann_ts = pd.Timestamp(ann_date).tz_localize(self.cfg.tz) + pd.Timedelta(hours=12)
                    else:
                        timing = "unknown"
                        ann_ts = pd.Timestamp(ann_date).tz_localize(self.cfg.tz) + pd.Timedelta(hours=10)

                    # fundamentals & surprise
                    eps_est = float(rng.normal(1.5, 0.6))
                    surprise_pct = float(rng.normal(0, 6))
                    eps_act = float(eps_est * (1 + surprise_pct / 100))
                    rev_est = float(s.mcap * 0.08)
                    rev_act = float(rev_est * (1 + rng.normal(0, 3) / 100))

                    # estimate timestamp: 3 days before announcement
                    est_ts = ann_ts - pd.Timedelta(days=3)

                    event_rows.append({
                        "event_id": f"E{len(event_rows):06d}",
                        "security_id": s.security_id,
                        "ticker": s.ticker,
                        "ticker_at_event": s.ticker,
                        "company_name": s.company_name,
                        "cik": int(s.cik),
                        "sector": s.sector,
                        "industry": s.industry,
                        "market_cap_at_event": float(s.mcap),
                        "cap_bucket": cap_b,
                        "fiscal_year": year,
                        "fiscal_quarter": q + 1,
                        "fiscal_period_end": fq_end,
                        "announcement_timestamp": ann_ts,
                        "announcement_timezone": self.cfg.tz,
                        "timing_class": timing,
                        "source": "mock",
                        "source_event_id": f"M-{s.security_id}-{year}-Q{q+1}",
                        "filing_accession_number": f"000-{s.cik:010d}-{year}{q+1:02d}",
                        "filing_timestamp": ann_ts,
                        "actual_eps": eps_act,
                        "estimated_eps": eps_est,
                        "actual_revenue": rev_act,
                        "estimated_revenue": rev_est,
                        "estimate_timestamp": est_ts,
                        "ingestion_timestamp": pd.Timestamp.utcnow().tz_convert("UTC"),
                        "data_quality_status": "ok",
                        "exclusion_reason": "",
                        "beta": float(s.beta),
                        "annual_vol": float(s.annual_vol),
                        "surprise_pct": surprise_pct,
                    })
                    est_rows.append({
                        "event_id": event_rows[-1]["event_id"],
                        "security_id": s.security_id,
                        "estimate_timestamp": est_ts,
                        "estimated_eps": eps_est,
                        "estimated_revenue": rev_est,
                        "n_analysts": int(rng.integers(3, 25)),
                    })

        events = pd.DataFrame(event_rows)
        estimates = pd.DataFrame(est_rows)

        # Plant signal: reaction CAR depends on surprise, segment, and decays over time.
        # We attach latent reaction & drift CAR for the LABEL step to read.
        gamma_year0 = start.year
        seg_mult_def = {"small_cap": 1.3, "mid_cap": 1.0, "large_cap": 0.7, "high_vol_regime": 1.2}
        seg_mult = {**seg_mult_def, **(seg_mult or {})}
        surprises = events["surprise_pct"].to_numpy()
        cap_mult = np.array([seg_mult.get(e, 1.0) for e in events["cap_bucket"]])
        vol_mult = np.array([seg_mult["high_vol_regime"] if v >= 0.4 else 1.0 for v in events["annual_vol"]])
        years_since = events["fiscal_year"].to_numpy() - gamma_year0
        gamma = (gamma_base * cap_mult * vol_mult) * np.maximum(0.05, 1 - decay * years_since)
        z_surprise = (surprises - surprises.mean()) / (surprises.std() + 1e-9)
        noise = rng.normal(0, noise_sd, size=len(events))
        reaction_car = gamma * z_surprise * 0.02 + noise * 0.005  # in decimal return
        drift_car = gamma * z_surprise * 0.02 * drift_mult + rng.normal(0, noise_sd, size=len(events)) * 0.003
        events["latent_reaction_car"] = reaction_car
        events["latent_drift_car_20"] = drift_car
        events["latent_realized_vol_20"] = events["annual_vol"].to_numpy() / np.sqrt(252) * (
            1 + 0.3 * z_surprise + rng.normal(0, 0.1, size=len(events))
        )
        return events, estimates

    def _make_filings(self, events: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed + 3)
        pos_words = ["strong", "growth", "record", "momentum", "expansion", "innovation", "robust"]
        neg_words = ["weak", "decline", "headwind", "pressure", "challenge", "restructuring", "uncertainty"]
        rows = []
        for _, e in events.iterrows():
            z = e["surprise_pct"] / 6.0
            tone = int(8 + 4 * z + rng.normal(0, 1.5))
            tone = max(0, tone)
            n_neg = int(8 - 4 * z + rng.normal(0, 1.5))
            n_neg = max(0, n_neg)
            body = " ".join([pos_words[i % len(pos_words)] for i in range(tone)] +
                            [neg_words[i % len(neg_words)] for i in range(n_neg)] +
                            ["revenue", "guidance", "margin", "outlook"])
            rows.append({
                "filing_accession_number": e["filing_accession_number"],
                "event_id": e["event_id"],
                "security_id": e["security_id"],
                "filing_type": "8-K",
                "filing_timestamp": e["filing_timestamp"],
                "section": "earnings_release",
                "text": body,
                "n_chars": len(body),
                "n_positive_words": tone,
                "n_negative_words": n_neg,
            })
        return pd.DataFrame(rows)

    def _security_master(self, stocks: pd.DataFrame) -> pd.DataFrame:
        from ..security_master.master import cap_bucket  # noqa: PLC0415
        rows = []
        for _, s in stocks.iterrows():
            rows.append({
                "security_id": s.security_id, "ticker": s.ticker, "cik": int(s.cik),
                "company_name": s.company_name, "sector": s.sector, "industry": s.industry,
                "market_cap_usd": float(s.mcap), "cap_bucket": cap_bucket(s.mcap),
                "valid_at": pd.Timestamp(s.listing_date), "listing_date": pd.Timestamp(s.listing_date),
                "delisting_date": pd.NaT,
            })
        return pd.DataFrame(rows)

    def _make_benchmark(self, sessions: pd.DatetimeIndex, ticker: str) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed + 4)
        rets = rng.normal(0.0002, 0.01, size=len(sessions))
        px = 200.0
        rows = []
        for j, s in enumerate(sessions):
            rows.append({"ticker": ticker, "date": pd.Timestamp(s).normalize(),
                         "open": px, "close": px * (1 + rets[j]), "high": px, "low": px, "volume": 0})
            px = px * (1 + rets[j])
        return pd.DataFrame(rows)

    def _make_sector_prices(self, sessions: pd.DatetimeIndex) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed + 5)
        rows = []
        for sector in SECTORS:
            rets = rng.normal(0.0001, 0.009, size=len(sessions))
            px = 100.0
            for j, s in enumerate(sessions):
                rows.append({"sector": sector, "ticker": "SECTOR_ETF",
                             "date": pd.Timestamp(s).normalize(),
                             "open": px, "close": px * (1 + rets[j]), "high": px, "low": px, "volume": 0})
                px = px * (1 + rets[j])
        return pd.DataFrame(rows)
