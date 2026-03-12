"""Integration test for stresstest __main__.py."""

import json

import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.stresstest.__main__ import main
from spx_ndx.models.spx_consensus.stresstest._helpers import NumpyEncoder


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Create a minimal workspace with synthetic signals parquet."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "output").mkdir()

    # Synthetic signal parquet (120 months, 2000-2009)
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2000-01-01", periods=n, freq="MS")
    signal = rng.choice([0.0, 1.0], size=n, p=[0.35, 0.65])
    buy_hold_returns = rng.normal(0.007, 0.04, n)
    cash_returns = np.full(n, 0.003)
    strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns

    df = pd.DataFrame({
        "signal": signal,
        "strategy_returns": strategy_returns,
        "buy_hold_returns": buy_hold_returns,
        "cash_returns": cash_returns,
    }, index=dates)
    parquet_path = str(tmp_path / "output" / "spx_consensus_signals.parquet")
    df.to_parquet(parquet_path)

    # Dataset with enough history for baseline signal construction
    # Start 24 months earlier so SMA10/12M mom have warmup data
    n_ds = n + 24
    ds_dates = pd.date_range("1998-01-01", periods=n_ds, freq="MS")
    spx_prices = np.cumprod(1 + rng.normal(0.007, 0.04, n_ds)) * 100
    ds = pd.DataFrame({
        "spx_close": spx_prices,
        "vix_close": rng.uniform(12, 35, n_ds),
        "ndx_close": np.cumprod(1 + rng.normal(0.008, 0.05, n_ds)) * 200,
        "rsi_14": rng.uniform(30, 80, n_ds),
        "cpi_yoy": rng.uniform(1.0, 5.0, n_ds),
        "cape_zscore": rng.normal(0, 1, n_ds),
        "spx_ema200_ratio": rng.normal(0.02, 0.05, n_ds),
        "composite_valuation": rng.normal(0.5, 0.5, n_ds),
    }, index=ds_dates)
    ds_path = str(tmp_path / "datas" / "dataset_monthly.parquet")
    (tmp_path / "datas").mkdir()
    ds.to_parquet(ds_path)

    return tmp_path, parquet_path, ds_path


def test_main_end_to_end(workspace, capsys):
    """Full run: loads data, runs all tests, exports JSON."""
    tmp_path, parquet_path, ds_path = workspace
    metrics = main([parquet_path, "--n", "50", "--seed", "42", "--dataset", ds_path])

    json_path = tmp_path / "output" / "spx_consensus_stress_metrics.json"
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)

    # All expected keys present
    expected_keys = {
        "meta", "baseline", "permutation", "bootstrap", "txcosts",
        "signal_noise", "return_noise", "montecarlo", "regimes",
        "vintage", "drawdowns", "rolling_alpha",
        "block_bootstrap", "correlation_bh",
        "trend_correlation", "baseline_correlation", "last_12",
        "decades", "rolling_rtr",
    }
    assert set(data.keys()) == expected_keys

    # Meta
    assert data["meta"]["n_periods"] == 120
    assert data["meta"]["periods_per_year"] == 12

    # Baseline
    assert "real_cagr" in data["baseline"]
    assert "buy_hold_cagr" in data["baseline"]

    # Permutation p-values in [0, 1]
    for k in ["p_cagr", "p_rtr", "p_sharpe", "p_max_drawdown"]:
        assert 0 <= data["permutation"][k] <= 1

    # Bootstrap CIs
    assert data["bootstrap"]["ci_cagr"][0] <= data["bootstrap"]["ci_cagr"][1]

    # Last 12
    assert len(data["last_12"]["dates"]) == 12

    # Stdout output
    captured = capsys.readouterr()
    assert "Loaded" in captured.out
    assert "Exported metrics to" in captured.out


def test_main_missing_dataset(workspace, capsys):
    """Run with nonexistent dataset -> fallback branches for VIX/cross-index/baseline."""
    tmp_path, parquet_path, _ = workspace
    metrics = main([parquet_path, "--n", "20", "--dataset", "/nonexistent/dataset.parquet"])

    # Should still complete with fallbacks
    assert "regimes" in metrics
    assert metrics["trend_correlation"]["correlations"] == {}
    assert metrics["baseline_correlation"]["correlations"] == {}


def test_numpy_encoder():
    """NumpyEncoder handles all NumPy types."""
    obj = {
        "bool": np.bool_(True),
        "int": np.int64(42),
        "float": np.float64(3.14),
        "float16": np.float16(2.5),  # not a native float -> hits np.floating branch
        "array": np.array([1, 2, 3]),
    }
    result = json.loads(json.dumps(obj, cls=NumpyEncoder))
    assert result["bool"] is True
    assert result["int"] == 42
    assert result["float"] == pytest.approx(3.14)
    assert result["float16"] == pytest.approx(2.5, abs=0.01)
    assert result["array"] == [1, 2, 3]


def test_numpy_encoder_unknown_type():
    """NumpyEncoder raises TypeError for unsupported types."""
    with pytest.raises(TypeError):
        json.dumps({"x": object()}, cls=NumpyEncoder)


def test_build_baseline_signals_nan():
    """_sig returns None when signal has NaN after ffill (df_index before dataset)."""
    from spx_ndx.models.spx_consensus.stresstest._helpers import build_baseline_signals

    # Dataset starts 2005, df_index starts 2000 -> SMA warmup produces NaN at start
    n_ds = 60
    ds_dates = pd.date_range("2005-01-01", periods=n_ds, freq="MS")
    ds = pd.DataFrame({
        "spx_close": np.cumprod(1 + np.full(n_ds, 0.01)) * 100,
    }, index=ds_dates)

    # df_index starts before dataset -> ffill can't fill leading NaNs
    df_index = pd.date_range("2000-01-01", periods=120, freq="MS")

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        ds.to_parquet(f.name)
        try:
            sigs = build_baseline_signals(f.name, df_index)
            # SMA 10m and 12M mom should be None (NaN before 2005) -> not in dict
            assert "SMA 10m" not in sigs
        finally:
            os.unlink(f.name)
