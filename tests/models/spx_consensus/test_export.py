import numpy as np
import pandas as pd
import pytest
from collections import Counter

import json
from pathlib import Path
from spx_ndx.models.spx_consensus.export import _build_parquet, _build_explain_json, export_results


@pytest.fixture
def mock_result():
    """Minimal result dict mimicking run_pipeline output."""
    dates = pd.date_range("2020-01", periods=6, freq="ME")
    return {
        "oos_returns": np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01]),
        "oos_signal": np.array([1.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
        "oos_dates": dates,
        "oos_agreement": np.array([3, 3, 1, 3, 1, 3]),
        "oos_deltas": np.random.rand(6, 2),
        "oos_raw_signals": np.array([[1, 0], [1, 1], [0, 0], [1, 1], [0, 1], [1, 0]], dtype=float),
        "oos_buy_hold": np.array([0.01, 0.02, -0.01, 0.03, -0.02, 0.01]),
        "oos_folds": np.array([1, 1, 1, 2, 2, 2]),
        "fold_results": [
            {"period": "2020-2021", "test_cagr": 0.08, "buy_hold_cagr": 0.10, "adapted_hps": {}},
            {"period": "2022-2023", "test_cagr": 0.05, "buy_hold_cagr": 0.03, "adapted_hps": {}},
        ],
        "cagr_thresholds": [0.07, 0.08, 0.09, 0.10, 0.11],
        "fold_structural_importance": [np.array([60.0, 40.0]), np.array([50.0, 50.0])],
        "last_structural_importance": np.array([55.0, 45.0]),
        "trader_counts": Counter({(2, ("A", "B")): 3, (1, ("A",)): 2}),
    }


class TestBuildParquet:
    def test_shape(self, mock_result):
        tbill = pd.Series(0.001, index=mock_result["oos_dates"])
        df = _build_parquet(mock_result, ["A", "B"], tbill)
        assert len(df) == 6
        assert "signal" in df.columns
        assert "strategy_returns" in df.columns
        assert "buy_hold_returns" in df.columns
        assert "cash_returns" in df.columns
        assert "agreement" in df.columns

    def test_per_signal_columns(self, mock_result):
        tbill = pd.Series(0.001, index=mock_result["oos_dates"])
        df = _build_parquet(mock_result, ["A", "B"], tbill)
        assert "raw_A" in df.columns
        assert "raw_B" in df.columns

    def test_no_dead_columns(self, mock_result):
        tbill = pd.Series(0.001, index=mock_result["oos_dates"])
        df = _build_parquet(mock_result, ["A", "B"], tbill)
        assert "fold" not in df.columns
        assert "delta_A" not in df.columns
        assert "delta_B" not in df.columns

    def test_index_matches_dates(self, mock_result):
        tbill = pd.Series(0.001, index=mock_result["oos_dates"])
        df = _build_parquet(mock_result, ["A", "B"], tbill)
        pd.testing.assert_index_equal(df.index, mock_result["oos_dates"])


class TestBuildExplainJson:
    def test_keys(self, mock_result):
        explain, global_percent = _build_explain_json(mock_result, ["A", "B"])
        assert "signal_names" in explain
        assert "n_pipelines" in explain
        assert "global_shapley_percent" in explain
        assert "signal_usage" in explain
        assert "fold_results" in explain

    def test_no_dead_keys(self, mock_result):
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        assert "ensemble_majority" not in explain
        assert "fold_structural_importance" not in explain

    def test_n_pipelines(self, mock_result):
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        assert explain["n_pipelines"] == 5

    def test_structural_importance(self, mock_result):
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        assert "structural_importance" in explain
        assert "formula" in explain
        assert "current_formula_value" in explain

    def test_global_percent_sums_to_100(self, mock_result):
        _, global_percent = _build_explain_json(mock_result, ["A", "B"])
        assert pytest.approx(global_percent.sum(), abs=0.1) == 100.0

    def test_signal_usage(self, mock_result):
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        assert explain["signal_usage"]["A"] == 5  # 3 from (A,B) + 2 from (A,)
        assert explain["signal_usage"]["B"] == 3  # 3 from (A,B)

    def test_fold_results(self, mock_result):
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        assert len(explain["fold_results"]) == 2
        assert explain["fold_results"][0]["period"] == "2020-2021"

    def test_no_duplicated_parquet_data(self, mock_result):
        """JSON should not duplicate data available in parquet."""
        explain, _ = _build_explain_json(mock_result, ["A", "B"])
        for key in ("last_12_raw_signals", "last_12_agreement", "last_12_dates", "last_fold_hps"):
            assert key not in explain


class TestExportResults:
    def test_writes_files(self, mock_result, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "output").mkdir()
        tbill = pd.Series(0.001, index=mock_result["oos_dates"])
        parquet_path, json_path, global_percent = export_results(mock_result, ["A", "B"], tbill)
        assert Path(parquet_path).exists()
        assert Path(json_path).exists()
        # Parquet readable
        df = pd.read_parquet(parquet_path)
        assert len(df) == 6
        # JSON readable
        with open(json_path) as f:
            data = json.load(f)
        assert "signal_names" in data
        # global_percent sums to 100
        assert pytest.approx(global_percent.sum(), abs=0.1) == 100.0
