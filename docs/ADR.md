# Architecture Decision Record

## ADR-001 — Default to a deterministic synthetic data provider

**Status:** Accepted (2026-07-27)

### Context
The spec requires paid point-in-time earnings/estimate/filing data (Compustat, IBES,
Refinitiv) plus a delisting-aware security master. None are available in this environment,
and credentials must not be embedded in source.

### Decision
1. Build the data layer behind a `DataProvider` adapter interface.
2. Ship a deterministic `MockProvider` as the default that produces a 100-stock,
   2015–2025 universe with quarterly events, estimates, filings text, and price paths,
   with a PLANTED signal (`gamma * normalize(surprise) * segment_factor + noise`) that
   decays over calendar time.
3. Implement real adapters for SEC EDGAR (PIT-correct filings + tickers) and yfinance
   (prices, flagged for survivorship/revision bias).
4. **Every empirical number produced by the default path is labeled SYNTHETIC** in
   artifacts, metrics, and the HTML report.
5. The deliverable is re-scoped as a **leakage-tested ML pipeline framework validated on
   synthetic data**, not an empirical study of US equities.

### Consequences
- No empirical claim about US equities is made by the default pipeline.
- The model-comparison ordering is partially predetermined by the synthetic signal shape
  (see ADR-004); the comparison is therefore validation of the pipeline, not a research result.
- A negative-control variant (signal γ = 0) is included to prove the pipeline does not
  manufacture alpha from noise.

### Alternatives considered
- Hard-coding plausible-looking empirical numbers: REJECTED — violates §1, §6, §18, §22.
- Treating yfinance as a real empirical anchor at headline level: REJECTED — survivorship
  bias gives an unknown positive bias to alpha and cannot support a defensible conclusion.

---

## ADR-002 — Text features OFF by default

**Status:** Accepted.

### Context
The default MockProvider emits synthetic filing text whose dictionary-tone correlates
mechanically with the planted surprise signal. Running text features by default would make
the headline result an artifact of that correlation rather than a validation of the text
pipeline.

### Decision
- Text features (dictionary tone, Δ-tone, TF-IDF) are gated behind
  `configs/features.yaml -> text.enabled: false`.
- The default pipeline runs structured-only.
- TF-IDF vocabulary is fit INSIDE each training fold by the model layer (no future documents).
- An optional embedding stub exists but is OFF the critical path.

---

## ADR-003 — yfinance is a smoke path, not an empirical anchor

**Status:** Accepted.

### Context
yfinance offers free public prices but (a) has no delisting security master, (b) applies
silent revisions to adjusted prices, and (c) reflects current (not historical) index
membership.

### Decision
- The `EdgarYFinanceProvider` adapter is implemented for completeness but is OFF by default.
- A CI test fails fast if any backtest universe ticker has a first-price-date after the
  prediction timestamp (IPO leakage).
- Any yfinance-derived number is reported only as a smoke check, never as a headline.

---

## ADR-004 — Pre-registered expected model ordering on synthetic data

**Status:** Accepted (per confirmation-agent requirement #10).

### Expected ordering (stated BEFORE running):
1. Tree models (LightGBM ≈ RandomForest) recover the planted non-linear, segment-dependent
   signal best.
2. Linear / ElasticNet recover the linear component of the signal.
3. SUE-decile and sign-of-surprise recover the surprise direction.
4. Unconditional mean and event-day-unconditional have no predictive power (except via drift).

Because the signal is planted, this ordering is largely predetermined. The model-comparison
table in the report therefore validates that the pipeline *can distinguish* models; it is
**not** an empirical finding about which model class wins on real earnings data.

---

## ADR-005 — Primary conclusion at the MEDIAN transaction cost

**Status:** Accepted (per confirmation-agent requirement #5).

The primary economic conclusion uses the median of the cost sweep (20 bps round trip),
not a single flat number. Cost sweep = {0, 5, 10, 20, 30, 50} bps RT. On synthetic data
without microstructure this is fictional, but the conservative anchor (20 bps) prevents
overstating net performance.

---

## ADR-006 — Reaction and drift are separate portfolios

**Status:** Accepted (per confirmation-agent requirement #4).

The reaction-horizon portfolio (exit at reaction_exit_session close) and the drift-horizon
portfolio (exit at drift_exit_session close) are reported as **distinct** backtest series
and are never mixed in a single return stream.
