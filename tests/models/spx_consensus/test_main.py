"""Integration test for __main__.py - runs the full CLI pipeline on frozen data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.__main__ import (
    main, _print_fold_detail, _format_fold_row, _format_transitions,
)

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture()
def workspace(tmp_path, monkeypatch):
    """Set up a workspace with dataset + config symlinked from fixtures."""
    (tmp_path / "datas").mkdir()
    (tmp_path / "datas" / "dataset_monthly.parquet").symlink_to(
        _FIXTURES / "dataset_monthly.parquet"
    )
    (tmp_path / "output").mkdir()
    config_src = _FIXTURES / "test_config.yaml"
    config_dst = tmp_path / "test_config.yaml"
    config_dst.symlink_to(config_src)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_main_end_to_end(workspace, capsys):
    """Full pipeline run: prints output, writes parquet+json, metrics are deterministic."""
    main(str(workspace / "test_config.yaml"))

    # --- Check exported files ---
    parquet_path = workspace / "output" / "spx_consensus_signals.parquet"
    json_path = workspace / "output" / "spx_consensus_explainability.json"
    assert parquet_path.exists()
    assert json_path.exists()

    df = pd.read_parquet(parquet_path)
    assert len(df) == 240  # 2 folds × 10 years × 12 months
    assert "signal" in df.columns
    assert "strategy_returns" in df.columns

    with open(json_path) as f:
        explain = json.load(f)
    assert explain["n_pipelines"] == 3
    assert len(explain["signal_names"]) == 10

    # --- Check stdout contains key sections ---
    out = capsys.readouterr().out
    assert "frequency" in out
    assert "signals" in out
    assert "Fold 1" in out
    assert "Fold 2" in out
    assert "RESULTS" in out
    assert "Exported" in out

    # --- Pin OOS metrics (deterministic with NUMBA_DISABLE_JIT=1 + frozen dataset) ---
    # These values come from the frozen dataset + test_config.yaml
    assert "5.9%" in out or "6.0%" in out  # strategy CAGR ~5.9%
    assert "15.3%" in out or "15.2%" in out  # strategy MaxDD ~15.3%


def test_result_none(workspace, monkeypatch, capsys):
    """run_pipeline returns None -> prints 'No results' and exits."""
    from spx_ndx.models.spx_consensus import __main__ as mod
    monkeypatch.setattr(mod, "run_pipeline", lambda *a, **kw: None)
    with pytest.raises(SystemExit, match="1"):
        main(str(workspace / "test_config.yaml"))
    out = capsys.readouterr().out
    assert "No results" in out


def test_all_failed_fold(capsys):
    """Fold with all_failed=True -> prints 'All thresholds failed' and returns."""
    fold = {
        "all_failed": True,
        "duration": 1.5,
        "fold": 1,
        "train_start": "2000",
        "train_end": "2009",
        "period": "2010-2011",
    }
    _print_fold_detail(fold, ["SMA", "VIX"])
    out = capsys.readouterr().out
    assert "All thresholds failed" in out


def test_format_fold_row_no_hps():
    """Fold row without adaptive HPs."""
    fold_result = {
        "fold": 1, "period": "2010-2011",
        "test_rtr": 0.85, "test_sharpe": 0.70, "test_cagr": 0.08,
        "test_max_drawdown": -0.15, "buy_hold_rtr": 0.60, "buy_hold_sharpe": 0.40,
        "buy_hold_cagr": 0.06, "buy_hold_max_drawdown": -0.50,
    }
    row = _format_fold_row(fold_result, [], [], lambda v, w: "")
    assert "2010-2011" in row
    assert "0.85" in row
    assert "8.0%" in row


def test_format_fold_row_with_hps():
    """Fold row with adaptive HPs."""
    fold_result = {
        "fold": 2, "period": "2012-2013",
        "test_rtr": 1.0, "test_sharpe": 0.9, "test_cagr": 0.10,
        "test_max_drawdown": -0.10, "buy_hold_rtr": 0.5, "buy_hold_sharpe": 0.3,
        "buy_hold_cagr": 0.04, "buy_hold_max_drawdown": -0.40,
        "adapted_hps": {"vote_threshold": 3},
    }
    row = _format_fold_row(fold_result, ["vote_threshold"], [4],
                        lambda v, w: f"{v!s:>{w}}")
    assert "2012-2013" in row
    assert "10.0%" in row


def test_format_transitions_empty():
    """No transitions -> 'No transitions'."""
    result = _format_transitions([])
    assert "No transitions" in result


def test_format_transitions_basic():
    """One transition -> formatted output."""
    tr = {
        "date": pd.Timestamp("2020-03-01"),
        "direction": "OUT->IN",
        "n_valid": 5,
        "agreement": 4,
        "changed": ["SMA", "VIX"],
        "drivers": ["SMA"],
        "votes": [0.8, 0.6, np.nan],
    }
    result = _format_transitions([tr])
    assert "2020-03" in result
    assert "IN" in result
    assert "4/5" in result
    assert "SMA" in result


def test_format_transitions_many_changed():
    """More than 6 changed signals -> '+N' suffix."""
    tr = {
        "date": pd.Timestamp("2020-01-01"),
        "direction": "IN->OUT",
        "n_valid": 10,
        "agreement": 3,
        "changed": ["A", "B", "C", "D", "E", "F", "G", "H"],
        "drivers": [],
        "votes": [0.5],
    }
    result = _format_transitions([tr])
    assert "+2" in result
    assert "(unanimous)" in result
