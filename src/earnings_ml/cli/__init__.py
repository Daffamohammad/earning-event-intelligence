"""Command-line interface for the earnings-ml pipeline."""
from __future__ import annotations

import json
import sys
import traceback

import click

from ..config import ensure_dirs, load_config
from ..logging import get_logger
from ..utils import seed_everything, write_json, write_parquet

log = get_logger(__name__)


def _cfg(ctx) -> object:
    return ctx.obj["cfg"]


def _validate_contract(name: str, df) -> None:
    """Enforce the pandera data contract for a pipeline table (hard error on violation).

    The contracts package lives at the repository root (not inside src/), so when the
    package is used outside the repo checkout the check is skipped with a warning.
    """
    try:
        from data_contracts.schemas import SCHEMAS  # noqa: PLC0415
    except ImportError:
        log.warning("data_contracts not importable; skipping %s contract validation", name)
        return
    SCHEMAS[name].validate(df)


@click.group()
@click.option("--config-dir", default=None, help="Path to configs/ directory")
@click.pass_context
def cli(ctx, config_dir):
    cfg = load_config(config_dir)
    ensure_dirs(cfg)
    seed_everything(cfg.seed)
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = cfg


@cli.command()
@click.pass_context
def inspect(ctx):
    """Show repository + config inspection summary."""
    cfg = _cfg(ctx)
    click.echo(json.dumps({
        "provider": cfg.provider,
        "seed": cfg.seed,
        "tz": cfg.tz,
        "universe": cfg.base.get("universe"),
        "config_hash": cfg.hash(),
        "artifacts_dir": str(cfg.artifacts_dir),
    }, indent=2))


@cli.command()
@click.pass_context
def ingest(ctx):
    """Run the data provider and persist raw tables to parquet."""
    cfg = _cfg(ctx)
    from ..ingestion.base import get_provider  # noqa: PLC0415
    out = get_provider(cfg).fetch()
    for name in ["events", "prices", "estimates", "filings"]:
        _validate_contract(name, getattr(out, name))
    paths = cfg.paths
    write_parquet(out.events, paths["events_parquet"])
    write_parquet(out.prices, paths["prices_parquet"])
    write_parquet(out.estimates, paths["estimates_parquet"])
    write_parquet(out.filings, paths["filings_parquet"])
    write_parquet(out.benchmark_prices, "artifacts/data/benchmark.parquet")
    write_parquet(out.sector_prices, "artifacts/data/sector_prices.parquet")
    write_parquet(out.security_master, "artifacts/data/security_master.parquet")
    write_json({"provider": out.provider, "is_synthetic": out.is_synthetic,
                "n_events": int(len(out.events)), "n_prices": int(len(out.prices)),
                "config_hash": cfg.hash()},
               cfg.artifacts_dir / "metrics/ingest_summary.json")
    click.echo(f"ingested {len(out.events)} events (provider={out.provider}, "
               f"synthetic={out.is_synthetic})")


@cli.command()
@click.pass_context
def build_events(ctx):
    """Build canonical event master with effective dates and PIT timestamps."""
    cfg = _cfg(ctx)
    from ..event_alignment.master import build_event_master  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    events = read_parquet(cfg.paths["events_parquet"])
    master = build_event_master(events, cfg)
    _validate_contract("event_master", master)
    write_parquet(master, "artifacts/data/event_master.parquet")
    n_primary = int((master["exclusion_reason"].fillna("") == "").sum())
    click.echo(f"event_master: {len(master)} events, {n_primary} primary")


@cli.command()
@click.pass_context
def validate_events(ctx):
    """Run data-quality + golden + PIT invariant checks on the event master."""
    cfg = _cfg(ctx)
    from ..event_alignment.master import validate_effective_dates  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    from ..validation.golden import evaluate_golden  # noqa: PLC0415
    master = read_parquet("artifacts/data/event_master.parquet")
    align_report = validate_effective_dates(master)
    bad = ((~align_report["is_session"]).sum()
           + (~align_report["amc_next_session_ok"]).sum()
           + (~align_report["bmo_same_session_ok"]).sum()
           + (~align_report["intraday_unknown_excluded_ok"]).sum())
    golden = evaluate_golden(cfg)
    write_json({
        "alignment_failures": int(bad),
        "golden": golden,
        "n_events": int(len(master)),
        "n_primary": int((master["exclusion_reason"].fillna("") == "").sum()),
        "exclusion_breakdown": master["exclusion_reason"].fillna("").replace("", "primary").value_counts().to_dict(),
    }, cfg.artifacts_dir / "metrics/event_validation.json")
    if golden.get("effective_date_accuracy", 1.0) < 0.98:
        click.echo(f"WARNING: golden eff-date accuracy {golden['effective_date_accuracy']:.3f} < 0.98", err=True)
    click.echo(f"alignment failures: {bad}; golden accuracy: {golden.get('effective_date_accuracy')}")


@cli.command()
@click.pass_context
def build_features(ctx):
    """Build structured (and optional text) features."""
    cfg = _cfg(ctx)
    from ..features.structured import build_structured_features  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    master = read_parquet("artifacts/data/event_master.parquet")
    prices = read_parquet(cfg.paths["prices_parquet"])
    bench = read_parquet("artifacts/data/benchmark.parquet")
    feats = build_structured_features(master, prices, bench, cfg)
    _validate_contract("features", feats)
    write_parquet(feats, cfg.paths["features_parquet"])
    click.echo(f"features: {len(feats)} rows, {len(feats.columns)} cols")

    if cfg.features.get("text", {}).get("enabled", False):
        from ..text_processing.features import build_text_features  # noqa: PLC0415
        filings = read_parquet(cfg.paths["filings_parquet"])
        text_feats = build_text_features(master, filings, cfg)
        if not text_feats.empty:
            write_parquet(text_feats, "artifacts/data/text_features.parquet")
            click.echo(f"text features: {len(text_feats)} rows")


@cli.command()
@click.pass_context
def validate_features(ctx):
    """Run PIT-invariant checks on features."""
    cfg = _cfg(ctx)
    from ..utils import read_parquet  # noqa: PLC0415
    from ..validation.invariants import check_feature_availability  # noqa: PLC0415
    feats = read_parquet(cfg.paths["features_parquet"])
    bad = check_feature_availability(feats, strict=True)
    write_json({
        "n_features_rows": int(len(feats)),
        "pit_violations": int(len(bad)),
        "violations_sample": bad.head(5).to_dict("records") if len(bad) else [],
    }, cfg.artifacts_dir / "metrics/feature_validation.json")
    if len(bad) > 0:
        click.echo(f"ERROR: {len(bad)} PIT violations found", err=True)
        sys.exit(1)
    click.echo(f"PIT OK: 0 violations across {len(feats)} rows")


@cli.command()
@click.pass_context
def build_labels(ctx):
    """Build labels (reaction CAR, drift CAR, realized vol)."""
    cfg = _cfg(ctx)
    from ..labels.car import attach_latent_labels, build_labels  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    master = read_parquet("artifacts/data/event_master.parquet")
    prices = read_parquet(cfg.paths["prices_parquet"])
    bench = read_parquet("artifacts/data/benchmark.parquet")
    sector = read_parquet("artifacts/data/sector_prices.parquet")
    labels = build_labels(master, prices, bench, sector, cfg)
    # If synthetic provider, attach planted labels (documented in report)
    if "latent_reaction_car" in master.columns:
        labels = attach_latent_labels(master, labels)
    _validate_contract("labels", labels)
    write_parquet(labels, cfg.paths["labels_parquet"])
    click.echo(f"labels: {len(labels)} rows")


@cli.command(name="train")
@click.option("--target", default="reaction_car", help="Label column to predict")
@click.pass_context
def train(ctx, target):
    """Run walk-forward training/prediction for all models."""
    cfg = _cfg(ctx)
    from ..evaluation.walkforward import run_walkforward  # noqa: PLC0415
    from ..models.zoo import make_all_models  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    feats = read_parquet(cfg.paths["features_parquet"])
    labels = read_parquet(cfg.paths["labels_parquet"])
    res = run_walkforward(feats, labels, cfg, models=make_all_models(), target=target,
                          predictions_dir=cfg.artifacts_dir / "predictions")
    write_json({"target": target, "pooled_metrics": res["pooled_metrics"],
                "folds": res["folds"], "n_fold_rows": int(len(res["fold_metrics"]))},
               cfg.artifacts_dir / f"metrics/train_{target}.json")
    click.echo(f"trained target={target}: {len(res['fold_metrics'])} fold-model rows")


@cli.command(name="train-drift")
@click.pass_context
def train_drift(ctx):
    """Convenience: train on the 20-day drift target."""
    ctx.invoke(train, target="drift_car_20")


@cli.command()
@click.option("--model", default="lightgbm")
@click.option("--horizon", default="drift", type=click.Choice(["reaction", "drift"]))
@click.pass_context
def backtest(ctx, model, horizon):
    """Run portfolio backtest with cost sweep."""
    cfg = _cfg(ctx)
    from ..backtest.portfolio import run_backtest  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    master = read_parquet("artifacts/data/event_master.parquet")
    prices = read_parquet(cfg.paths["prices_parquet"])
    preds = read_parquet(cfg.artifacts_dir / "predictions/predictions.parquet")
    mp = preds[preds["model"] == model].copy()
    res = run_backtest(master, mp, prices, cfg, horizon=horizon)
    write_json(res, cfg.artifacts_dir / f"backtest/{model}_{horizon}.json")
    click.echo(f"backtest model={model} horizon={horizon}: "
               f"L/S net @ {res.get('primary_cost_bps')}bps = "
               f"{res.get('primary', {}).get('long_short_net')}")


@cli.command()
@click.option("--target", default="reaction_car")
@click.pass_context
def ablate(ctx, target):
    """Run ablation + robustness suite."""
    cfg = _cfg(ctx)
    from ..evaluation.ablation import run_ablations, run_robustness  # noqa: PLC0415
    from ..utils import read_parquet  # noqa: PLC0415
    feats = read_parquet(cfg.paths["features_parquet"])
    labels = read_parquet(cfg.paths["labels_parquet"])
    abl = run_ablations(feats, labels, cfg, target=target)
    rob = run_robustness(feats, labels, cfg, target=target)
    write_json({"ablations": abl, "robustness": rob, "target": target},
               cfg.artifacts_dir / f"metrics/ablation_{target}.json")
    click.echo(f"ablations: {len(abl)}; robustness: {len(rob)}")


@cli.command()
@click.pass_context
def report(ctx):
    """Generate the HTML research report from stored artifacts."""
    cfg = _cfg(ctx)
    from ..reporting.html import generate_report  # noqa: PLC0415
    out = generate_report(cfg)
    click.echo(f"report written to {out}")


@cli.command(name="run-all")
@click.pass_context
def run_all(ctx):
    """Execute the full reproducible pipeline in order."""
    steps = [
        ("inspect", lambda: ctx.invoke(inspect)),
        ("ingest", lambda: ctx.invoke(ingest)),
        ("build-events", lambda: ctx.invoke(build_events)),
        ("validate-events", lambda: ctx.invoke(validate_events)),
        ("build-features", lambda: ctx.invoke(build_features)),
        ("validate-features", lambda: ctx.invoke(validate_features)),
        ("build-labels", lambda: ctx.invoke(build_labels)),
        ("train", lambda: ctx.invoke(train, target="reaction_car")),
        ("ablate", lambda: ctx.invoke(ablate, target="reaction_car")),
        ("backtest", lambda: ctx.invoke(backtest, model="lightgbm", horizon="reaction")),
        ("backtest-drift", lambda: ctx.invoke(backtest, model="lightgbm", horizon="drift")),
        ("report", lambda: ctx.invoke(report)),
    ]
    for name, fn in steps:
        click.echo(f"\n=== {name} ===")
        try:
            fn()
        except (SystemExit, click.exceptions.Exit):
            raise
        except Exception:
            log.error("step %s failed:\n%s", name, traceback.format_exc())
            click.echo(f"FAIL: {name}", err=True)
            sys.exit(2)


# Allow `python -m earnings_ml.cli` invocation
if __name__ == "__main__":
    cli()
