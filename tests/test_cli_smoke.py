"""CLI smoke tests."""
from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from earnings_ml.cli import cli


def test_inspect_command():
    runner = CliRunner()
    r = runner.invoke(cli, ["inspect"])
    assert r.exit_code == 0, r.output
    assert "provider" in r.output
    assert "mock" in r.output


def test_cli_help_lists_all_commands():
    runner = CliRunner()
    r = runner.invoke(cli, ["--help"])
    assert r.exit_code == 0
    for cmd in ["inspect", "ingest", "build-events", "validate-events", "build-features",
                "validate-features", "build-labels", "train", "backtest", "report", "run-all"]:
        assert cmd in r.output, f"missing CLI command: {cmd}"
