import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.signals import (
    threshold_signal, band_signal, compute_signals, build_sig_matrix,
    _make_name, HANDLERS,
)


class TestThresholdSignal:
    """Tests for threshold_signal."""

    def test_below(self):
        series = pd.Series([1.0, 2.0, 3.0, 0.5])
        result = threshold_signal(series, 2.5, "below")
        expected = pd.Series([1.0, 1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_above(self):
        series = pd.Series([1.0, 2.0, 3.0, 0.5])
        result = threshold_signal(series, 1.5, "above")
        expected = pd.Series([0.0, 1.0, 1.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_preserved(self):
        series = pd.Series([1.0, np.nan, 3.0])
        result = threshold_signal(series, 2.0, "below")
        assert result.iloc[0] == 1
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 0

    def test_sma_then_threshold(self):
        closes = pd.Series([10, 11, 12, 13, 14, 15], dtype=float)
        sma_values = closes.rolling(3).mean()
        diff = closes - sma_values
        result = threshold_signal(diff, 0, "above")
        assert result.dropna().iloc[-1] == 1

    def test_rsi_then_threshold(self):
        closes = pd.Series(range(100, 150), dtype=float)
        delta = closes.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        relative_strength = gain / loss
        rsi_values = 100 - 100 / (1 + relative_strength)
        result = threshold_signal(rsi_values, 70, "below")
        assert result.iloc[-1] == 0


class TestBandSignal:
    """Tests for band_signal."""

    def test_in_band(self):
        series = pd.Series([10, 50, 80, 40])
        result = band_signal(series, lower=30, upper=70)
        expected = pd.Series([0.0, 1.0, 0.0, 1.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_nan_preserved(self):
        series = pd.Series([50, np.nan, 40])
        result = band_signal(series, lower=30, upper=70)
        assert result.iloc[0] == 1
        assert np.isnan(result.iloc[1])
        assert result.iloc[2] == 1


# --- T3: _make_name ---

class TestMakeName:
    def test_basic_replacement(self):
        assert _make_name("VIX<X", 20) == "VIX<20"

    def test_last_x_replaced(self):
        assert _make_name("RSI 30-X", 70) == "RSI 30-70"

    def test_none_returns_template(self):
        assert _make_name("SomeTemplate", None) == "SomeTemplate"

    def test_sma_template(self):
        assert _make_name("SMA Xm", 10) == "SMA 10m"

    def test_multiple_x_replaces_last(self):
        assert _make_name("X>X", 5) == "X>5"


# --- T3: HANDLERS ---

class TestHandlers:
    @pytest.fixture
    def dataset(self):
        index = pd.date_range("2000-01", periods=5, freq="ME")
        return pd.DataFrame({
            "vix_close": [25.0, 15.0, 10.0, 30.0, 20.0],
            "rsi_14": [40.0, 60.0, 80.0, 20.0, 50.0],
            "macd_hist": [0.5, -0.3, 0.1, -0.5, 0.2],
            "macd_line": [1.0, -1.0, 0.5, -0.5, 0.0],
            "cape_zscore": [0.5, -0.5, 1.0, -1.0, 0.0],
            "sma_10m": [100.0, 102.0, 104.0, 106.0, 108.0],
            "spx_ema200_ratio": [2.0, -1.0, 3.0, 0.5, 1.5],
        }, index=index)

    @pytest.fixture
    def closes(self, dataset):
        return pd.Series([105.0, 101.0, 110.0, 103.0, 115.0],
                         index=dataset.index)

    def test_vix_below(self, dataset, closes):
        handler = HANDLERS["VIX<X"]
        result = handler(dataset, closes, 20)
        assert result.iloc[0] == 0  # 25 >= 20
        assert result.iloc[1] == 1  # 15 < 20
        assert result.iloc[2] == 1  # 10 < 20

    def test_rsi_below(self, dataset, closes):
        handler = HANDLERS["RSI<X"]
        result = handler(dataset, closes, 50)
        assert result.iloc[0] == 1  # 40 < 50
        assert result.iloc[1] == 0  # 60 >= 50

    def test_rsi_above(self, dataset, closes):
        handler = HANDLERS["RSI>X"]
        result = handler(dataset, closes, 50)
        assert result.iloc[0] == 0  # 40 <= 50
        assert result.iloc[1] == 1  # 60 > 50

    def test_rsi_band(self, dataset, closes):
        handler = HANDLERS["RSI 30-X"]
        result = handler(dataset, closes, 70)
        assert result.iloc[0] == 1  # 30 < 40 < 70
        assert result.iloc[2] == 0  # 80 >= 70

    def test_vp_val(self, dataset, closes):
        """VP>VAL handler returns threshold signal when column exists."""
        dataset["vp_val_6m"] = [100.0, 102.0, 110.0, 90.0, 108.0]
        handler = HANDLERS["VP>VAL Xm"]
        result = handler(dataset, closes, 6)
        # closes - vp_val: 5, -1, 0, 13, 7
        assert result.iloc[0] == 1  # 105 > 100
        assert result.iloc[1] == 0  # 101 < 102

    def test_vp_val_missing_column(self, dataset, closes):
        handler = HANDLERS["VP>VAL Xm"]
        result = handler(dataset, closes, 99)
        assert result is None

    def test_sma(self, dataset, closes):
        handler = HANDLERS["SMA Xm"]
        result = handler(dataset, closes, 10)
        # closes - sma_10m: 5, -1, 6, -3, 7
        assert result.iloc[0] == 1  # 5 > 0
        assert result.iloc[1] == 0  # -1 <= 0

    def test_ema200_with_transform(self, dataset, closes):
        handler = HANDLERS["EMA200>X"]
        result = handler(dataset, closes, 1)  # transform: 1/100 = 0.01
        assert result.iloc[0] == 1  # 2.0 > 0.01
        assert result.iloc[1] == 0  # -1.0 <= 0.01

    def test_missing_column_returns_none(self, closes):
        handler = HANDLERS["VIX<X"]
        empty_dataset = pd.DataFrame(index=closes.index)
        assert handler(empty_dataset, closes, 20) is None

    def test_macd_hist(self, dataset, closes):
        handler = HANDLERS["MACDhist>X"]
        result = handler(dataset, closes, 0)
        assert result.iloc[0] == 1  # 0.5 > 0
        assert result.iloc[1] == 0  # -0.3 <= 0


# --- T1: compute_signals ---

class TestComputeSignals:
    def test_basic(self):
        index = pd.date_range("2000-01", periods=4, freq="ME")
        dataset = pd.DataFrame({
            "vix_close": [25.0, 15.0, 10.0, 30.0],
            "rsi_14": [40.0, 60.0, 80.0, 20.0],
        }, index=index)
        closes = pd.Series([100, 110, 120, 130], dtype=float, index=index)
        indicators = {"VIX<X": [20, 25], "RSI<X": [50]}
        result = compute_signals(dataset, closes, indicators)
        assert "VIX<20" in result
        assert "VIX<25" in result
        assert "RSI<50" in result
        assert len(result) == 3

    def test_unknown_template_skipped(self):
        index = pd.date_range("2000-01", periods=3, freq="ME")
        dataset = pd.DataFrame({"vix_close": [10, 20, 30]}, index=index)
        closes = pd.Series([100, 110, 120], dtype=float, index=index)
        result = compute_signals(dataset, closes, {"UNKNOWN<X": [10]})
        assert result == {}

    def test_missing_column_skipped(self):
        index = pd.date_range("2000-01", periods=3, freq="ME")
        dataset = pd.DataFrame({"other": [1, 2, 3]}, index=index)
        closes = pd.Series([100, 110, 120], dtype=float, index=index)
        result = compute_signals(dataset, closes, {"VIX<X": [20]})
        assert result == {}

    def test_output_is_binary(self):
        index = pd.date_range("2000-01", periods=4, freq="ME")
        dataset = pd.DataFrame({"vix_close": [15.0, 25.0, 10.0, 30.0]}, index=index)
        closes = pd.Series([100, 110, 120, 130], dtype=float, index=index)
        result = compute_signals(dataset, closes, {"VIX<X": [20]})
        values = result["VIX<20"].dropna().values
        assert set(values).issubset({0.0, 1.0})


# --- T2: build_sig_matrix ---

class TestBuildSigMatrix:
    def test_shape(self):
        index = pd.date_range("2000-01", periods=5, freq="ME")
        signals = {
            "A": pd.Series([1, 0, 1, 0, 1], dtype=float, index=index),
            "B": pd.Series([0, 1, 0, 1, 0], dtype=float, index=index),
        }
        matrix = build_sig_matrix(signals, ["A", "B"], index)
        assert matrix.shape == (5, 2)
        assert matrix.dtype == np.float64

    def test_shift_by_one(self):
        """Signal is shifted by 1 - uses previous month's value."""
        index = pd.date_range("2000-01", periods=4, freq="ME")
        signals = {"A": pd.Series([0, 0, 1, 1], dtype=float, index=index)}
        matrix = build_sig_matrix(signals, ["A"], index)
        # shift(1): [NaN, 0, 0, 1] -> fillna(1): [1, 0, 0, 1]
        np.testing.assert_array_equal(matrix[:, 0], [1, 0, 0, 1])

    def test_nan_filled_with_one(self):
        """NaN values become 1 (default = invested)."""
        index = pd.date_range("2000-01", periods=3, freq="ME")
        signals = {"A": pd.Series([np.nan, np.nan, 0], dtype=float, index=index)}
        matrix = build_sig_matrix(signals, ["A"], index)
        # ffill on [NaN, NaN, 0] -> [NaN, NaN, 0], fillna(1) -> [1, 1, 0]
        # shift(1) -> [NaN, 1, 1], fillna(1) -> [1, 1, 1]
        np.testing.assert_array_equal(matrix[:, 0], [1, 1, 1])

    def test_ffill(self):
        """Forward-fill gaps before shifting."""
        index = pd.date_range("2000-01", periods=5, freq="ME")
        signals = {"A": pd.Series([1, np.nan, np.nan, 0, np.nan], dtype=float, index=index)}
        matrix = build_sig_matrix(signals, ["A"], index)
        # ffill: [1, 1, 1, 0, 0], fillna(1): same, shift(1): [NaN, 1, 1, 1, 0], fillna(1): [1, 1, 1, 1, 0]
        np.testing.assert_array_equal(matrix[:, 0], [1, 1, 1, 1, 0])

    def test_reindex_to_subset(self):
        """Can reindex to a subset of the signal's index."""
        full_index = pd.date_range("2000-01", periods=6, freq="ME")
        sub_index = full_index[2:5]
        signals = {"A": pd.Series([0, 0, 1, 1, 0, 0], dtype=float, index=full_index)}
        matrix = build_sig_matrix(signals, ["A"], sub_index)
        assert matrix.shape == (3, 1)
