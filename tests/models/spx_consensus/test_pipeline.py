import numpy as np
import pandas as pd
import pytest

from types import SimpleNamespace

from spx_ndx.models.spx_consensus.pipeline import (
    generate_folds, detect_transitions, aggregate_oos,
    _group_weights, slice_data, _median_combo, _passes_floor,
    _score_ensemble, _failed_fold_result, SplitData, _Context,
)


class TestGenerateFolds:
    def test_basic(self):
        folds = generate_folds(1993, 1999, 2, 2004)
        assert len(folds) == 3
        assert folds[0] == ("1993-01-01", "1998-12-31", "1999-01-01", "2000-12-31")
        assert folds[2] == ("1993-01-01", "2002-12-31", "2003-01-01", "2004-12-31")

    def test_empty_when_no_room(self):
        assert generate_folds(1993, 2010, 2, 2005) == []

    def test_single_fold(self):
        folds = generate_folds(1993, 2000, 5, 2004)
        assert len(folds) == 1
        assert folds[0] == ("1993-01-01", "1999-12-31", "2000-01-01", "2004-12-31")

    def test_exact_boundary(self):
        assert len(generate_folds(1993, 1999, 2, 2000)) == 1

    def test_one_year_short(self):
        assert generate_folds(1993, 1999, 2, 1999) == []


class TestDetectTransitions:
    def test_no_transitions(self):
        final = np.ones(5)
        sig_test = np.ones((5, 3))
        deltas = np.zeros((5, 3))
        agreement = np.full(5, 3.0)
        votes = np.full((5, 3), 0.8)
        idx = pd.date_range("2000-01", periods=5, freq="ME")
        result = detect_transitions(final, sig_test, ["A", "B", "C"], deltas,
                                    agreement, votes, None, idx, 0)
        assert result == []

    def test_intra_fold_transition(self):
        final = np.array([1.0, 1.0, 0.0, 0.0, 1.0])
        sig_test = np.ones((5, 2))
        sig_test[2, 0] = 0
        deltas = np.zeros((5, 2))
        deltas[2, 0] = 0.5
        agreement = np.array([3.0, 3.0, 1.0, 1.0, 3.0])
        votes = np.full((5, 3), 0.8)
        idx = pd.date_range("2000-01", periods=5, freq="ME")
        result = detect_transitions(final, sig_test, ["A", "B"], deltas,
                                    agreement, votes, None, idx, 0)
        assert len(result) == 2
        assert result[0]["direction"] == "IN->OUT"
        assert result[1]["direction"] == "OUT->IN"
        assert not result[0]["fold_boundary"]

    def test_fold_boundary_transition(self):
        final = np.array([0.0, 0.0, 1.0])
        sig_test = np.ones((3, 2))
        deltas = np.zeros((3, 2))
        deltas[0, 0] = 0.3
        agreement = np.array([1.0, 1.0, 3.0])
        votes = np.full((3, 3), 0.8)
        idx = pd.date_range("2000-01", periods=3, freq="ME")
        result = detect_transitions(final, sig_test, ["A", "B"], deltas,
                                    agreement, votes, 1.0, idx, 1)
        assert len(result) == 2
        assert result[0]["fold_boundary"] is True
        assert result[0]["direction"] == "IN->OUT"

    def test_fold_boundary_out_to_in(self):
        """Fold boundary with OUT->IN transition."""
        final = np.array([1.0, 1.0])
        sig_test = np.ones((2, 2))
        deltas = np.zeros((2, 2))
        deltas[0, 0] = 0.2
        agreement = np.array([3.0, 3.0])
        votes = np.full((2, 3), 0.8)
        idx = pd.date_range("2000-01", periods=2, freq="ME")
        result = detect_transitions(final, sig_test, ["A", "B"], deltas,
                                    agreement, votes, 0.0, idx, 0)
        assert len(result) == 1
        assert result[0]["direction"] == "OUT->IN"
        assert result[0]["fold_boundary"] is True
        # changed should list ON signals
        assert any("=ON" in c for c in result[0]["changed"])

    def test_fold_boundary_many_signals_truncated(self):
        """More than 5 signals -> truncated with +N."""
        n = 8
        final = np.array([0.0, 0.0])
        sig_test = np.zeros((2, n))  # all OFF
        deltas = np.zeros((2, n))
        agreement = np.array([1.0, 1.0])
        votes = np.full((2, 3), 0.8)
        idx = pd.date_range("2000-01", periods=2, freq="ME")
        names = [f"S{i}" for i in range(n)]
        result = detect_transitions(final, sig_test, names, deltas,
                                    agreement, votes, 1.0, idx, 0)
        assert len(result) == 1
        assert result[0]["direction"] == "IN->OUT"
        # Should truncate at 5 and add +3
        assert any("+3" in c for c in result[0]["changed"])

    def test_n_valid_derived_from_votes(self):
        """n_valid is derived from pipe_votes (non-NaN columns)."""
        final = np.array([1.0, 0.0])
        sig_test = np.ones((2, 2))
        deltas = np.zeros((2, 2))
        agreement = np.array([2.0, 0.0])
        # 3 pipelines but 1 is NaN -> n_valid=2
        votes = np.array([[0.8, 0.8, np.nan], [0.2, 0.2, np.nan]])
        idx = pd.date_range("2000-01", periods=2, freq="ME")
        result = detect_transitions(final, sig_test, ["A", "B"], deltas,
                                    agreement, votes, None, idx, 0)
        assert len(result) == 1
        assert result[0]["n_valid"] == 2


class TestFailedFoldResult:
    def test_structure(self):
        returns_test = np.array([0.01, -0.01, 0.02])
        signal_test = np.ones((3, 2))
        base_result = {"fold": 1, "period": "2000-2001"}
        context = SimpleNamespace(n_signals=2, periods_per_year=12.0)
        result = _failed_fold_result(
            returns_test, signal_test, context, 0, pd.date_range("2000-01", periods=3, freq="ME"),
            base_result, np.zeros(3), [None], 1.5
        )
        assert result["fold_result"]["all_failed"] is True
        assert result["fold_result"]["duration"] == 1.5
        assert len(result["oos_returns"]) == 3
        assert result["oos_folds"].tolist() == [1, 1, 1]
        assert result["fold_result"]["test_cagr"] == result["fold_result"]["buy_hold_cagr"]


class TestAggregateOos:
    def test_basic(self):
        dates1 = pd.date_range("2000-01", periods=3, freq="ME")
        dates2 = pd.date_range("2000-04", periods=3, freq="ME")
        tbill = pd.Series(0.001, index=pd.date_range("1999-01", periods=24, freq="ME"))

        accum = {
            "oos_returns": [np.array([0.01, 0.02, -0.01]), np.array([0.03, -0.02, 0.01])],
            "oos_signal": [np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])],
            "oos_buy_hold": [np.array([0.01, 0.02, -0.01]), np.array([0.03, -0.02, 0.01])],
            "oos_dates": [dates1, dates2],
            "oos_agreement": [np.array([3, 3, 1]), np.array([3, 1, 3])],
            "oos_deltas": [np.zeros((3, 2)), np.zeros((3, 2))],
            "oos_raw_signals": [np.ones((3, 2)), np.ones((3, 2))],
            "oos_folds": [np.array([1, 1, 1]), np.array([2, 2, 2])],
        }

        arrays, metrics_oos, metrics_buy_hold_oos = aggregate_oos(accum, tbill, 12.0)
        assert len(arrays["oos_returns"]) == 6
        assert len(arrays["oos_dates"]) == 6
        assert arrays["oos_folds"].tolist() == [1, 1, 1, 2, 2, 2]
        assert "cagr" in metrics_oos
        assert "cagr" in metrics_buy_hold_oos


# --- T4: _group_weights ---

class TestGroupWeights:
    def _make_groups(self, rtrs, cagrs, stabs):
        """Helper: build top_groups list from metric arrays."""
        return [(r, c, s, 2, (0, 1)) for r, c, s in zip(rtrs, cagrs, stabs)]

    def test_equal(self):
        groups = self._make_groups([1.0, 2.0, 3.0], [0.1, 0.2, 0.3], [0.5, 0.6, 0.7])
        weights = _group_weights(groups, "equal")
        np.testing.assert_allclose(weights, [1/3, 1/3, 1/3])

    def test_cagr_weighted(self):
        groups = self._make_groups([1.0, 1.0], [0.10, 0.30], [0.5, 0.5])
        weights = _group_weights(groups, "cagr_weighted")
        assert weights[1] > weights[0]  # higher cagr -> higher weight
        assert pytest.approx(weights.sum()) == 1.0

    def test_rtr_weighted(self):
        groups = self._make_groups([1.0, 3.0], [0.10, 0.10], [0.5, 0.5])
        weights = _group_weights(groups, "rtr_weighted")
        assert weights[1] > weights[0]
        assert pytest.approx(weights.sum()) == 1.0

    def test_stability_weighted(self):
        groups = self._make_groups([1.0, 1.0], [0.10, 0.10], [0.3, 0.9])
        weights = _group_weights(groups, "stability_weighted")
        assert weights[1] > weights[0]
        assert pytest.approx(weights.sum()) == 1.0

    def test_zero_sum_fallback(self):
        """All negative raw weights -> fallback to equal."""
        groups = self._make_groups([-1.0, -2.0], [0.10, 0.10], [0.5, 0.5])
        weights = _group_weights(groups, "rtr_weighted")
        np.testing.assert_allclose(weights, [0.5, 0.5])

    def test_unknown_aggregation(self):
        groups = self._make_groups([1.0], [0.1], [0.5])
        with pytest.raises(ValueError, match="Unknown aggregation"):
            _group_weights(groups, "unknown_mode")


# --- T5: slice_data ---

class TestSliceData:
    def test_basic(self):
        idx = pd.date_range("2000-01-01", periods=12, freq="ME")
        returns = pd.Series(np.arange(12, dtype=float), index=idx)
        tbill = pd.Series(0.001, index=idx)
        returns_slice, cash_slice = slice_data(returns, tbill, "2000-03-01", "2000-08-31")
        assert len(returns_slice) == 6
        assert len(cash_slice) == 6
        assert (cash_slice == 0.001).all()

    def test_empty_range(self):
        idx = pd.date_range("2000-01-01", periods=6, freq="ME")
        returns = pd.Series(np.ones(6), index=idx)
        tbill = pd.Series(0.001, index=idx)
        returns_slice, cash_slice = slice_data(returns, tbill, "2010-01-01", "2010-12-31")
        assert len(returns_slice) == 0

    def test_tbill_ffill(self):
        """T-bill values are forward-filled when missing."""
        idx = pd.date_range("2000-01-01", periods=4, freq="ME")
        returns = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
        tbill = pd.Series([0.005, np.nan, np.nan, 0.006], index=idx)
        _, cash_slice = slice_data(returns, tbill, "2000-01-01", "2000-12-31")
        assert cash_slice.iloc[1] == 0.005  # forward-filled


# --- T6: _median_combo ---

class TestMedianCombo:
    def test_single_combo(self):
        result = _median_combo([{"a": 1, "b": 2}], ["a", "b"])
        assert result == {"a": 1, "b": 2}

    def test_odd_number(self):
        combos = [{"x": 1}, {"x": 3}, {"x": 5}]
        result = _median_combo(combos, ["x"])
        assert result["x"] == 3  # median of [1, 3, 5]

    def test_even_number(self):
        combos = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
        result = _median_combo(combos, ["x"])
        assert result["x"] == 3  # index 2 of sorted [1,2,3,4]

    def test_multiple_keys(self):
        combos = [{"a": 10, "b": 1}, {"a": 20, "b": 3}, {"a": 30, "b": 2}]
        result = _median_combo(combos, ["a", "b"])
        assert result["a"] == 20
        assert result["b"] == 2


# --- T7: _passes_floor ---

class TestPassesFloor:
    def test_no_floor(self):
        assert _passes_floor({"stability": 0.5}, None) is True

    def test_empty_metrics(self):
        assert _passes_floor({}, {"stability": 0.5}) is True

    def test_passes(self):
        assert _passes_floor({"stability": 0.8}, {"stability": 0.5}) is True

    def test_fails(self):
        assert _passes_floor({"stability": 0.3}, {"stability": 0.5}) is False

    def test_exact_threshold(self):
        assert _passes_floor({"stability": 0.5}, {"stability": 0.5}) is True


# --- T8: _score_ensemble ---

def _make_trader(min_votes, combo, rtr=1.0, cagr=0.1, stability=0.9):
    return (rtr, cagr, stability, min_votes, combo)

class TestScoreEnsemble:
    def test_all_none_returns_neg_inf(self):
        validation = SplitData(np.ones((5, 2)), np.ones(5), np.zeros(5))
        score, metrics = _score_ensemble([None, None], validation, 0.5, 12.0)
        assert score == (-np.inf, -np.inf)
        assert metrics == {}

    def test_positive_returns(self):
        """All signals ON, positive returns -> positive CAGR."""
        np.random.seed(42)
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = (traders, groups, weights)
        validation = SplitData(np.ones((24, 2)), np.random.normal(0.01, 0.02, 24), np.zeros(24))
        score, metrics = _score_ensemble([model], validation, 0.5, 12.0)
        assert score[0] > 0  # CAGR > 0
        assert "cagr" in metrics
        assert "rtr" in metrics

    def test_score_tuple_matches_metrics(self):
        """Score tuple = (metrics['cagr'], metrics['rtr'])."""
        traders = [_make_trader(1, (0,))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = (traders, groups, weights)
        validation = SplitData(np.ones((12, 1)), np.full(12, 0.02), np.zeros(12))
        score, metrics = _score_ensemble([model], validation, 0.5, 12.0)
        assert score == (metrics["cagr"], metrics["rtr"])
