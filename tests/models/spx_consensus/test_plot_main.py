"""Integration test for the plot module - runs generate_all on real fixture data."""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.plot.generate import generate_all
from spx_ndx.models.spx_consensus.plot.__main__ import main as plot_main

_FIXTURES = Path(__file__).parent / "fixtures"
_SIGNALS = _FIXTURES / "spx_consensus_signals.parquet"
_METRICS = _FIXTURES / "spx_consensus_stress_metrics.json"
_EXPLAIN = _FIXTURES / "spx_consensus_explainability.json"


@pytest.fixture(scope="module")
def plot_workspace(tmp_path_factory):
    """Set up workspace with real fixtures: signals parquet, metrics JSON, explain JSON."""
    tmp_path = tmp_path_factory.mktemp("plot_test")

    # Copy fixtures into workspace output/ and datas/
    (tmp_path / "output").mkdir()
    (tmp_path / "datas").mkdir()

    shutil.copy(_SIGNALS, tmp_path / "output" / "spx_consensus_signals.parquet")
    shutil.copy(_EXPLAIN, tmp_path / "output" / "spx_consensus_explainability.json")

    # Copy real dataset for plots that read datas/dataset_monthly.parquet
    _project_root = _FIXTURES.parent.parent.parent
    real_dataset = _project_root / "datas" / "dataset_monthly.parquet"
    if real_dataset.exists():
        shutil.copy(real_dataset, tmp_path / "datas" / "dataset_monthly.parquet")

    with open(_METRICS) as f:
        metrics = json.load(f)
    parquet_path = str(tmp_path / "output" / "spx_consensus_signals.parquet")
    metrics["meta"]["parquet"] = parquet_path
    metrics_path = str(tmp_path / "output" / "spx_consensus_stress_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)

    explain_path = str(tmp_path / "output" / "spx_consensus_explainability.json")
    df = pd.read_parquet(parquet_path)

    return metrics, df, "consensus", tmp_path, explain_path, metrics_path


def test_generate_all(plot_workspace):
    """generate_all produces all plots without errors."""
    metrics, df, label, tmp_path, explain_path, _ = plot_workspace

    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        saved = generate_all(metrics, df, label, explain_path=explain_path)
    finally:
        os.chdir(old_cwd)

    assert len(saved) >= 10
    for p in saved:
        if isinstance(p, str):
            assert (tmp_path / p).exists(), f"Missing: {p}"
        elif isinstance(p, list):
            for pp in p:
                assert (tmp_path / pp).exists(), f"Missing: {pp}"


def test_plot_main_cli(plot_workspace, monkeypatch):
    """plot __main__.main() runs end-to-end."""
    _, _, _, tmp_path, explain_path, metrics_path = plot_workspace

    monkeypatch.chdir(tmp_path)
    import sys
    monkeypatch.setattr(sys, "argv", [
        "plot", metrics_path, "--explain", explain_path,
    ])
    plot_main()
