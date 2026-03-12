"""Edge-case tests for plot/plots.py - covers guard branches and exception paths."""

import json
import os
import tempfile

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.plot.generate import generate_all
from spx_ndx.models.spx_consensus.plot.cross_index import plot_cross_index
from spx_ndx.models.spx_consensus.plot.decades import plot_decades
from spx_ndx.models.spx_consensus.plot.explain import plot_explain
from spx_ndx.models.spx_consensus.plot.folds_cagr import plot_folds_cagr
from spx_ndx.models.spx_consensus.plot.return_noise import plot_return_noise
from spx_ndx.models.spx_consensus.plot.rolling_rtr import plot_rolling_rtr
from spx_ndx.models.spx_consensus.plot.txcosts import plot_txcosts
from spx_ndx.models.spx_consensus.plot.vintage_dca import plot_vintage_dca, _irr


@pytest.fixture(autouse=True)
def close_all_figures():
    yield
    plt.close("all")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(n=60):
    """Minimal DataFrame with signal/returns."""
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    return pd.DataFrame({
        "signal": np.ones(n),
        "strategy_returns": np.full(n, 0.005),
        "buy_hold_returns": np.full(n, 0.005),
        "cash_returns": np.full(n, 0.003),
    }, index=dates)


# ── plot_txcosts - line 127: breakeven annotation ────────────────────────────

def test_txcosts_breakeven_annotation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()

    metrics = {
        "baseline": {"buy_hold_cagr": 0.06},
        "txcosts": {
            "costs_bps": [0, 10, 30, 50],
            "adjusted_cagrs": [0.09, 0.08, 0.05, 0.03],
            "n_trades": 50,
            "breakeven": 35,  # triggers line 127
        },
    }
    df = _make_df()
    fig, path = plot_txcosts(metrics, df, "test")
    assert path is not None


# ── plot_return_noise - line 248: all noise levels below B&H ─────────────────

def test_return_noise_all_below_bh(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()

    metrics = {
        "baseline": {"real_cagr": 0.04, "buy_hold_cagr": 0.10},
        "return_noise": {
            "noise_mults": [0.5, 1.0, 2.0],
            "results": {
                "0.5": {"cagrs": [0.03, 0.04, 0.02]},
                "1.0": {"cagrs": [0.02, 0.01, 0.03]},
                "2.0": {"cagrs": [0.01, 0.00, 0.02]},
            },
        },
    }
    df = _make_df()
    fig, path = plot_return_noise(metrics, df, "test")
    assert path is not None


# ── _irr - lines 683-684: brentq ValueError ─────────────────────────────────

def test_irr_no_solution():
    """Cashflows with no IRR solution -> returns NaN."""
    # All positive cashflows -> no root exists
    result = _irr([100, 100, 100])
    assert np.isnan(result)


# ── plot_explain - lines 1266, 1268: guards ──────────────────────────────────

def test_explain_empty_last12():
    metrics = {"last_12": {}}
    assert plot_explain(metrics, _make_df(), "test") == (None, None)


def test_explain_missing_file():
    metrics = {"last_12": {"dates": ["2024-01"], "signals": [1]}}
    assert plot_explain(metrics, _make_df(), "test",
                        explain_path="/nonexistent.json") == (None, None)


# ── plot_explain - line 1371: formula empty -> fallback text ──────────────────

def test_explain_no_formula(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "datas").mkdir()

    # Minimal dataset for SPX chart
    n = 12
    dates = pd.date_range("2024-01-01", periods=n, freq="MS")
    ds = pd.DataFrame({"spx_close": np.linspace(4000, 5000, n)}, index=dates)
    ds.to_parquet(tmp_path / "datas" / "dataset_monthly.parquet")

    sig_names = ["SMA", "VIX"]
    df = pd.DataFrame({
        "signal": np.ones(n),
        "strategy_returns": np.full(n, 0.005),
        "buy_hold_returns": np.full(n, 0.005),
        "cash_returns": np.full(n, 0.003),
        "agreement": np.full(n, 3),
        "raw_SMA": np.ones(n),
        "raw_VIX": np.ones(n),
    }, index=dates)

    explain = {
        "signal_names": sig_names,
        "n_pipelines": 5,
        "structural_importance": [0.6, 0.4],
        "formula": [],  # empty -> triggers line 1371
        "current_formula_value": None,
    }
    explain_path = str(tmp_path / "output" / "explain.json")
    with open(explain_path, "w") as f:
        json.dump(explain, f)

    metrics = {
        "meta": {"periods_per_year": 12},
        "last_12": {
            "dates": [d.strftime("%Y-%m") for d in dates],
            "signals": [1] * n,
            "strategy_returns": [0.005] * n,
        },
    }
    fig, path = plot_explain(metrics, df, "test", explain_path=explain_path)
    assert path is not None


# ── plot_folds_cagr - lines 1595, 1601: missing explain / empty folds ───────

def test_folds_cagr_missing_file():
    metrics = {}
    assert plot_folds_cagr(metrics, _make_df(), "test",
                           explain_path="/nonexistent.json") == (None, None)


def test_folds_cagr_empty_folds(tmp_path):
    explain = {"fold_results": []}
    path = str(tmp_path / "explain.json")
    with open(path, "w") as f:
        json.dump(explain, f)
    metrics = {}
    assert plot_folds_cagr(metrics, _make_df(), "test", explain_path=path) == (None, None)


# ── plot_decades - lines 1696, 1765: empty / no Sharpe ──────────────────────

def test_decades_empty():
    metrics = {"decades": {}}
    assert plot_decades(metrics, _make_df(), "test") == (None, None)


def test_decades_missing_key():
    metrics = {}
    assert plot_decades(metrics, _make_df(), "test") == (None, None)


def test_decades_no_sharpe(tmp_path, monkeypatch):
    """Decades with data but without Sharpe keys -> returns empty."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()

    metrics = {
        "decades": {
            "2000s": {
                "strategy_cagr": 0.08, "buy_hold_cagr": 0.05,
                "strategy_rtr": 1.2, "buy_hold_rtr": 0.5,
                "strategy_max_drawdown": -0.15, "buy_hold_max_drawdown": -0.50,
            },
            "2010s": {
                "strategy_cagr": 0.10, "buy_hold_cagr": 0.12,
                "strategy_rtr": 1.4, "buy_hold_rtr": 0.9,
                "strategy_max_drawdown": -0.10, "buy_hold_max_drawdown": -0.30,
            },
        },
    }
    fig, path = plot_decades(metrics, _make_df(), "test")
    assert fig is None and path is None  # no Sharpe -> returns None


# ── plot_rolling_rtr - line 1830: missing key ───────────────────────────────

def test_rolling_rtr_missing():
    metrics = {}
    assert plot_rolling_rtr(metrics, _make_df(), "test") == (None, None)


def test_rolling_rtr_none():
    metrics = {"rolling_rtr": None}
    assert plot_rolling_rtr(metrics, _make_df(), "test") == (None, None)


# ── generate_all internals - lines 2070, 2081-2083: _add(None), _safe() ─────

def test_safe_catches_exception(tmp_path, monkeypatch, capsys):
    """_safe wrapper catches exceptions and prints to stderr."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()

    # Minimal metrics that will make individual plot functions crash
    metrics = {
        "meta": {"periods_per_year": 12, "n_periods": 60, "frequency": "monthly",
                 "parquet": str(tmp_path / "dummy.parquet")},
        "baseline": {"real_cagr": 0.08, "buy_hold_cagr": 0.06,
                     "real_rtr": 1.0, "real_sharpe": 0.7},
        "permutation": None,  # will crash plot_permutation -> triggers _safe
    }
    # Minimal df
    n = 60
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    df = pd.DataFrame({
        "signal": np.ones(n),
        "strategy_returns": np.full(n, 0.005),
        "buy_hold_returns": np.full(n, 0.005),
        "cash_returns": np.full(n, 0.003),
    }, index=dates)
    df.to_parquet(tmp_path / "dummy.parquet")

    # generate_all should not crash - _safe catches individual failures
    saved = generate_all(metrics, df, "test")
    # Some plots skipped -> stderr has [SKIP] messages
    err = capsys.readouterr().err
    assert "[SKIP]" in err


# ── plot_cross_index - lines 371, 379: missing index / column ────────────────

def test_cross_index_missing_index(tmp_path, monkeypatch):
    """Dataset has no NDX/MSCI columns -> only SPX plotted."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "datas").mkdir()

    n = 60
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    ds = pd.DataFrame({"spx_close": np.linspace(1000, 2000, n)}, index=dates)
    ds.to_parquet(tmp_path / "datas" / "dataset_monthly.parquet")

    df = _make_df(n)
    signal = df["signal"].values
    strategy_returns = df["strategy_returns"].values
    buy_hold_returns = df["buy_hold_returns"].values
    cash_returns = df["cash_returns"].values

    # No cross_index in metrics - computed on the fly from dataset
    metrics = {"meta": {"frequency": "monthly", "periods_per_year": 12}}
    paths = plot_cross_index(metrics, df, "test", signal, strategy_returns, buy_hold_returns, cash_returns)
    assert len(paths) >= 1  # at least SPX


def test_cross_index_missing_column(tmp_path, monkeypatch):
    """Dataset has no ndx_close -> NDX skipped."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "datas").mkdir()

    n = 60
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    # Dataset has spx_close but NOT ndx_close
    ds = pd.DataFrame({"spx_close": np.linspace(1000, 2000, n)}, index=dates)
    ds.to_parquet(tmp_path / "datas" / "dataset_monthly.parquet")

    df = _make_df(n)
    signal = df["signal"].values
    strategy_returns = df["strategy_returns"].values
    buy_hold_returns = df["buy_hold_returns"].values
    cash_returns = df["cash_returns"].values

    # No cross_index in metrics - computed on the fly from dataset
    metrics = {"meta": {"frequency": "monthly", "periods_per_year": 12}}
    paths = plot_cross_index(metrics, df, "test", signal, strategy_returns, buy_hold_returns, cash_returns)
    # NDX skipped (no column), SPX plotted
    assert len(paths) >= 1


# ── plot_vintage_dca - lines 785-786: brentq ValueError ─────────────────────

def test_vintage_dca_brentq_valueerror(tmp_path, monkeypatch):
    """DCA vintage where brentq raises ValueError -> breakeven=None (lines 785-786)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output").mkdir()
    (tmp_path / "datas").mkdir()

    n = 24
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    buy_hold_returns = np.full(n, 0.03)
    signal = np.zeros(n)  # always in cash -> strategy loses -> enters brentq path
    cash_returns = np.full(n, 0.001)
    strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns

    df = pd.DataFrame({
        "signal": signal,
        "strategy_returns": strategy_returns,
        "buy_hold_returns": buy_hold_returns,
        "cash_returns": cash_returns,
    }, index=dates)

    ds = pd.DataFrame({"tbill_rate": np.full(n, 3.0)}, index=dates)
    ds.to_parquet(tmp_path / "datas" / "dataset_monthly.parquet")

    metrics = {
        "meta": {"frequency": "monthly", "periods_per_year": 12},
        "vintage": {
            "years": {"2000": {"wins": False, "cagr_strategy": 0.01, "cagr_buy_hold": 0.30}},
            "wins": 0, "total": 1, "percent": 0,
        },
    }

    # Monkeypatch brentq at source so `from scipy.optimize import brentq` gets it
    import scipy.optimize
    _real_brentq = scipy.optimize.brentq

    def _brentq_raises(*args, **kwargs):
        raise ValueError("f(a) and f(b) must have different signs")
    monkeypatch.setattr(scipy.optimize, "brentq", _brentq_raises)

    paths = plot_vintage_dca(metrics, df, "test", signal, buy_hold_returns, cash_returns)
