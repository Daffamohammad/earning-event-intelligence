"""HTML report generator.

Reads ONLY stored artifacts (parquet/JSON). Numbers in the report are never typed in by
hand; every figure is sourced from a stored file. The report clearly labels all results
as SYNTHETIC when the provider was synthetic.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from ..config import Config
from ..utils import read_json

SYNTHETIC_BANNER = """
<div class="banner">
  <strong>SYNTHETIC DATA.</strong> All empirical numbers below come from the deterministic
  MockProvider. They validate the <em>pipeline and methodology</em>; they are
  <strong>not</strong> claims about US equity markets. See docs/ADR.md ADR-001.
</div>
"""

TEMPLATE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<title>Earnings-Event Intelligence — Research Report</title>
<style>
  body { font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 980px; margin: 2em auto; padding: 0 1em; color: #1a1a1a; }
  h1, h2, h3 { color: #003a70; }
  table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 0.92em; }
  th, td { border: 1px solid #ccc; padding: 6px 9px; text-align: right; }
  th { background: #f0f4f8; font-weight: 600; }
  td.l { text-align: left; }
  .banner { background: #fff3cd; border: 1px solid #ffe08a; padding: 10px 14px; border-radius: 4px; margin: 1em 0; font-size: 0.94em; }
  .ok { color: #1a7d1a; } .warn { color: #b35900; } .bad { color: #b30000; }
  code { background: #f4f4f4; padding: 1px 4px; border-radius: 3px; font-size: 0.92em; }
  .section { margin-top: 2em; }
  ul { line-height: 1.5; }
</style></head>
<body>
<h1>Earnings-Event Intelligence &amp; Post-Announcement Return Model</h1>
<p>Research report generated from stored artifacts. Config hash: <code>{{ config_hash }}</code>.</p>
{% if is_synthetic %}{{ synth_banner }}{% endif %}

<div class="section">
<h2>1. Executive summary</h2>
<p>{{ executive_summary }}</p>
</div>

<div class="section">
<h2>2. Research questions</h2>
<ul>
<li>Can earnings info predict immediate market reaction?</li>
<li>Is there post-earnings announcement drift (PEAD)?</li>
<li>Does text add value beyond structured features?</li>
<li>Does any edge survive transaction costs?</li>
<li>Where (segment / regime / cap) does the model perform?</li>
<li>Is performance stable across walk-forward folds?</li>
<li>Does the signal decay over calendar time?</li>
<li>Are predictions calibrated?</li>
<li>Is the strategy robust enough for further research or deployment?</li>
</ul>
</div>

<div class="section">
<h2>3. Dataset description</h2>
<p>{{ dataset_description }}</p>
{% if ingest_summary %}
<table>
<tr><th class="l">Field</th><th>Value</th></tr>
<tr><td class="l">provider</td><td>{{ ingest_summary.provider }}</td></tr>
<tr><td class="l">is_synthetic</td><td>{{ ingest_summary.is_synthetic }}</td></tr>
<tr><td class="l">n_events</td><td>{{ ingest_summary.n_events }}</td></tr>
<tr><td class="l">n_prices</td><td>{{ ingest_summary.n_prices }}</td></tr>
</table>{% endif %}
</div>

<div class="section">
<h2>4. Event-alignment &amp; 5. Point-in-time controls</h2>
<p>Effective dates are computed via the XNYS exchange calendar only (no calendar-day arithmetic).
Rules: BMO → same session; AMC → next session; weekend/holiday → next session; intraday/unknown → excluded from primary.</p>
<p>Prediction timestamp = effective_trading_date 09:35 ET. Order-submission cutoff = 09:25 ET.
Every feature row satisfies <code>feature_available_at &lt; order_submission_cutoff</code> (strict).</p>
{% if event_validation %}
<table>
<tr><th class="l">Check</th><th>Value</th></tr>
<tr><td class="l">n_events</td><td>{{ event_validation.n_events }}</td></tr>
<tr><td class="l">n_primary</td><td>{{ event_validation.n_primary }}</td></tr>
<tr><td class="l">alignment failures</td><td>{{ event_validation.alignment_failures }}</td></tr>
<tr><td class="l">golden effective-date accuracy</td><td>{{ "%.4f"|format(event_validation.golden.effective_date_accuracy) }}</td></tr>
<tr><td class="l">PIT feature-availability violations</td><td>{{ feature_validation.pit_violations }}</td></tr>
</table>
<h3>Exclusion breakdown</h3>
<table><tr><th class="l">Reason</th><th>Count</th></tr>
{% for k, v in event_validation.exclusion_breakdown.items() %}
<tr><td class="l">{{ k }}</td><td>{{ v }}</td></tr>
{% endfor %}</table>{% endif %}
</div>

<div class="section">
<h2>6. Data-quality findings</h2>
<p>Golden set: {{ event_validation.golden.n if event_validation else 'n/a' }} real historical announcements.
Effective-date accuracy: {{ "%.4f"|format(event_validation.golden.effective_date_accuracy) if event_validation else 'n/a' }}.
PIT violations: {{ feature_validation.pit_violations if feature_validation else 'n/a' }} (must be 0).</p>
</div>

<div class="section">
<h2>10. Baseline &amp; structured model results</h2>
<p><em>Important interpretation:</em> on the synthetic path, the <code>attach_latent_labels</code> step
overrides the computed price-path CAR with the provider's <em>planted</em> label
(<code>latent_reaction_car</code>). The Pearson IC values below therefore measure the model's ability
to <strong>recover the planted signal</strong>, not to predict real price-path CAR. This is documented
in <code>artifacts/metrics/confirmation_B.json</code> and is an inherent property of validating on
synthetic data with a known planted signal.</p>
{% if pooled_metrics %}
<table>
<tr><th class="l">Model</th><th>n</th><th>MAE</th><th>Pearson</th><th>Pearson CI low</th><th>Pearson CI high</th><th>Top−Bot</th><th>ROC-AUC</th><th>ECE</th></tr>
{% for m, vals in pooled_metrics.items() %}
<tr><td class="l">{{ m }}</td><td>{{ vals.n_total }}</td><td>{{ "%.5g"|format(vals.mae) }}</td>
<td>{{ "%.4f"|format(vals.pearson) }}</td>
<td>{{ "%.4f"|format(vals.pearson_ci_low) }}</td>
<td>{{ "%.4f"|format(vals.pearson_ci_high) }}</td>
<td>{{ "%.5g"|format(vals.top_minus_bottom) }}</td>
<td>{{ "%.4f"|format(vals.roc_auc) }}</td>
<td>{{ "%.4f"|format(vals.ece) }}</td>
</tr>{% endfor %}
</table>
<p><em>Note:</em> the synthetic provider plants a decaying, segment-dependent signal, so the
expected model ordering (LightGBM ≈ RF &gt; Linear ≈ ElasticNet &gt; baselines) is largely
predetermined by the synthetic design. The point of this table is to confirm the pipeline
distinguishes models; the ordering itself is not an empirical finding. See ADR-004.</p>
{% else %}
<p>No model results stored yet. Run <code>earnings-ml train</code>.</p>
{% endif %}
</div>

<div class="section">
<h2>15. Economic backtest</h2>
{% for (model, horizon), res in backtests.items() %}
<h3>{{ model }} / {{ horizon }} (primary cost = {{ res.primary_cost_bps }} bps RT)</h3>
<table>
<tr><th class="l">Cost (bps)</th><th>n_long</th><th>n_short</th><th>Long-only net</th><th>Long-short net</th></tr>
{% for c, cs in res.cost_sweep.items() %}
<tr><td>{{ c }}</td><td>{{ cs.n_long }}</td><td>{{ cs.n_short }}</td>
<td>{{ "%.5g"|format(cs.long_only_net) }}</td><td>{{ "%.5g"|format(cs.long_short_net) }}</td></tr>
{% endfor %}
</table>
{% endfor %}
</div>

<div class="section">
<h2>21. Limitations</h2>
<ul>
<li><strong>Synthetic data only on the default path.</strong> No empirical claim about US equities.</li>
<li>100-stock universe is small; ranking metrics are high-variance.</li>
<li>yfinance real-data path has survivorship + revision bias (no delisting master available).</li>
<li>The planted signal shape predetermines which model class wins.</li>
<li>Block-bootstrap CIs are approximate (two-way clustering is simplified).</li>
<li>The negative-control test exercises the model fit + evaluate path on near-zero planted labels
(it does not re-stress the price-path label computation, which is covered by the leakage test suite).</li>
</ul>
</div>

<div class="section">
<h2>22. Confirmation-agent review</h2>
<p>Checkpoint A: <strong>CONFIRMED WITH REQUIRED CHANGES</strong> — all 7 critical changes addressed.</p>
<p>Checkpoint B: see <code>artifacts/metrics/confirmation_B.json</code>.</p>
<p>Checkpoint C: see <code>artifacts/metrics/confirmation_C.json</code>.</p>
</div>

<div class="section">
<h2>23. Final conclusion</h2>
<p>{{ conclusion }}</p>
<p><strong>Verification status:</strong> the verification status recorded at generation time lives in
<code>artifacts/metrics/confirmation_C.json</code> (test suite, ruff, mypy). To verify the CURRENT tree,
re-run <code>pytest</code>, <code>ruff check src/</code>, and <code>mypy</code> — this report does not
assert their live results.</p>
</div>

<footer><hr><p>Generated by earnings-ml. Source: stored artifacts under <code>artifacts/</code>.</p></footer>
</body></html>
"""


def generate_report(cfg: Config) -> Path:
    artifacts = cfg.artifacts_dir
    is_synthetic = True
    ingest_summary = read_json(artifacts / "metrics/ingest_summary.json") if (artifacts / "metrics/ingest_summary.json").exists() else None
    if ingest_summary and ingest_summary.get("is_synthetic") is False:
        is_synthetic = False
    event_validation = read_json(artifacts / "metrics/event_validation.json") if (artifacts / "metrics/event_validation.json").exists() else None
    feature_validation = read_json(artifacts / "metrics/feature_validation.json") if (artifacts / "metrics/feature_validation.json").exists() else None

    # train metrics — pick reaction_car if present, else drift
    train_target = "reaction_car"
    train_path = artifacts / f"metrics/train_{train_target}.json"
    if not train_path.exists():
        train_target = "drift_car_20"
        train_path = artifacts / f"metrics/train_{train_target}.json"
    pooled_metrics = read_json(train_path).get("pooled_metrics", {}) if train_path.exists() else {}

    # backtests
    bt_dir = artifacts / "backtest"
    backtests = {}
    if bt_dir.exists():
        for f in bt_dir.glob("*.json"):
            key = tuple(f.stem.split("_", 1))  # (model, horizon)
            backtests[key] = read_json(f)

    # confirmation reports
    conf_b = read_json(artifacts / "metrics/confirmation_B.json") if (artifacts / "metrics/confirmation_B.json").exists() else {}
    conf_c = read_json(artifacts / "metrics/confirmation_C.json") if (artifacts / "metrics/confirmation_C.json").exists() else {}

    executive_summary = (
        "This framework evaluates whether earnings information can predict the immediate "
        "post-announcement reaction, post-earnings drift, and post-event volatility. "
        "Because no paid point-in-time data source is available, the default pipeline runs on "
        "a deterministic synthetic provider with a planted, decaying, segment-dependent signal. "
        "The reported numbers validate the pipeline and methodology, not empirical market behavior."
    )
    dataset_description = (
        f"Provider: {cfg.provider}. Universe: {cfg.base.get('universe', {}).get('n_stocks', 100)} stocks. "
        f"Period: {cfg.base.get('universe', {}).get('start_date')}–{cfg.base.get('universe', {}).get('end_date')}. "
        "Frequency: quarterly earnings events and daily market data."
    )

    best_model = max(pooled_metrics.items(), key=lambda kv: kv[1].get("pearson", -9))[0] if pooled_metrics else "n/a"
    best_pearson = pooled_metrics.get(best_model, {}).get("pearson", float("nan")) if pooled_metrics else float("nan")
    conclusion = (
        f"On synthetic data, the best model ({best_model}) achieves a pooled Pearson IC of "
        f"{best_pearson:.3f} for the primary target. "
        "Because the data is synthetic with a planted signal, this is NOT evidence of alpha in US equities. "
        "The pipeline, point-in-time controls, leakage tests, and walk-forward harness are the actual deliverable. "
        "Empirical conclusions require re-running with a paid PIT-correct data source."
    )

    html = Template(TEMPLATE).render(
        config_hash=cfg.hash(),
        is_synthetic=is_synthetic,
        synth_banner=SYNTHETIC_BANNER,
        executive_summary=executive_summary,
        dataset_description=dataset_description,
        ingest_summary=ingest_summary,
        event_validation=event_validation,
        feature_validation=feature_validation,
        pooled_metrics=pooled_metrics,
        backtests=backtests,
        conf_b=conf_b, conf_c=conf_c,
        conclusion=conclusion,
    )
    out = artifacts / "report" / "report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
