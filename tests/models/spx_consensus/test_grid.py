import numpy as np
import pytest
from spx_ndx.models.spx_consensus.grid import make_configs, grid_search, _eval_all, warmup

class TestMakeConfigs:
    def test_basic(self):
        configs = make_configs(3, 2, 2)
        assert configs.shape[1] == 3  # min_votes + 2 slots
        assert len(configs) > 0

    def test_empty(self):
        configs = make_configs(1, 2, 3)
        assert len(configs) == 0

    def test_counts(self):
        # n_items=4, size=2: C(4,2)=6 combos, each with 2 min_votes values = 12
        configs = make_configs(4, 2, 2)
        assert len(configs) == 12

    def test_padding(self):
        configs = make_configs(3, 2, 3)
        # max_size=3, so shape[1] = 1+3 = 4
        assert configs.shape[1] == 4
        # size=2 rows should have -1 padding in last slot
        size2_rows = configs[configs[:, 3] == -1]
        assert len(size2_rows) > 0

class TestGridSearch:
    def test_basic(self):
        np.random.seed(42)
        matrix = np.random.rand(24, 3).astype(np.float64)
        returns = np.random.normal(0.01, 0.04, 24).astype(np.float64)
        cash = np.full(24, 0.003, dtype=np.float64)
        configs = make_configs(3, 2, 2)
        results, n_total, n_pass = grid_search(
            matrix, returns, cash, configs, top_n=5,
            periods_per_year=12.0, min_rtr=-999, min_cagr=-999
        )
        assert len(results) <= 5
        assert n_total == len(configs)

    def test_empty_configs(self):
        configs = np.empty((0, 3), dtype=np.int64)
        results, n_total, n_pass = grid_search(
            np.ones((5, 2)), np.ones(5), np.ones(5),
            configs, 10, 12.0, 0, 0
        )
        assert results == []
        assert n_total == 0

    def test_all_pass_fewer_than_top_n(self):
        """When all configs pass and n_pass <= top_n, return all."""
        np.random.seed(42)
        matrix = np.random.rand(24, 3).astype(np.float64)
        returns = np.random.normal(0.01, 0.04, 24).astype(np.float64)
        cash = np.full(24, 0.003, dtype=np.float64)
        configs = make_configs(3, 2, 2)  # 12 configs
        results, n_total, n_pass = grid_search(
            matrix, returns, cash, configs, top_n=999,
            periods_per_year=12.0, min_rtr=-999, min_cagr=-999
        )
        assert len(results) == n_pass
        assert n_pass == n_total  # all pass with -999 thresholds

    def test_strict_filter(self):
        np.random.seed(42)
        matrix = np.random.rand(24, 3).astype(np.float64)
        returns = np.random.normal(0.001, 0.04, 24).astype(np.float64)
        cash = np.full(24, 0.003, dtype=np.float64)
        configs = make_configs(3, 2, 2)
        results, n_total, n_pass = grid_search(
            matrix, returns, cash, configs, top_n=5,
            periods_per_year=12.0, min_rtr=999, min_cagr=999
        )
        assert results == []
        assert n_pass == 0


class TestWarmup:
    def test_runs_without_error(self):
        warmup()  # should not raise


class TestEvalAll:
    """Tests for _eval_all (Numba-accelerated metric computation)."""

    def test_output_shapes(self):
        matrix = np.ones((12, 3), dtype=np.float64)
        returns = np.full(12, 0.01, dtype=np.float64)
        cash = np.full(12, 0.003, dtype=np.float64)
        configs = make_configs(3, 2, 2)
        rtrs, cagrs, stabs = _eval_all(matrix, returns, cash, configs, 12.0)
        assert len(rtrs) == len(configs)
        assert len(cagrs) == len(configs)
        assert len(stabs) == len(configs)

    def test_all_invested_positive_returns(self):
        """All signals=1, positive returns -> positive CAGR."""
        matrix = np.ones((24, 2), dtype=np.float64)
        returns = np.full(24, 0.01, dtype=np.float64)
        cash = np.zeros(24, dtype=np.float64)
        # Single config: min_votes=1, combo=(0,1)
        configs = np.array([[1, 0, 1]], dtype=np.int64)
        rtrs, cagrs, stabs = _eval_all(matrix, returns, cash, configs, 12.0)
        assert cagrs[0] > 0

    def test_all_cash(self):
        """All signals=0 -> earns cash rate, not market return."""
        matrix = np.zeros((24, 2), dtype=np.float64)
        returns = np.full(24, 0.05, dtype=np.float64)  # high market
        cash = np.full(24, 0.001, dtype=np.float64)  # low cash
        configs = np.array([[1, 0, 1]], dtype=np.int64)
        rtrs, cagrs, stabs = _eval_all(matrix, returns, cash, configs, 12.0)
        # CAGR should be close to cash rate, not market rate
        assert cagrs[0] < 0.05

    def test_total_loss(self):
        """Returns < -100% -> cum <= 0 -> cagr=-1, log_cum=-50, stab=1."""
        matrix = np.ones((1, 2), dtype=np.float64)
        returns = np.array([-2.0], dtype=np.float64)  # -200% -> cum = -1
        cash = np.zeros(1, dtype=np.float64)
        configs = np.array([[1, 0, 1]], dtype=np.int64)
        rtrs, cagrs, stabs = _eval_all(matrix, returns, cash, configs, 12.0)
        assert cagrs[0] == -1.0
        assert stabs[0] == 1.0

class TestSortBy:
    """Test that sort_by parameter controls ranking order."""

    def _make_data(self):
        np.random.seed(99)
        matrix = np.random.rand(36, 4).astype(np.float64)
        returns = np.random.normal(0.008, 0.04, 36).astype(np.float64)
        cash = np.full(36, 0.002, dtype=np.float64)
        configs = make_configs(4, 2, 2)
        return matrix, returns, cash, configs

    def test_sort_by_default(self):
        """Default sort_by should match stability,cagr."""
        matrix, returns, cash, configs = self._make_data()
        r_default, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999
        )
        r_explicit, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999,
            sort_by=("stability", "cagr")
        )
        for a, b in zip(r_default, r_explicit):
            assert a[4] == b[4]

    def test_sort_by_cagr_first(self):
        """sort_by=(cagr, stability) should rank by cagr descending."""
        matrix, returns, cash, configs = self._make_data()
        results, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999,
            sort_by=("cagr", "stability")
        )
        cagrs = [r[1] for r in results]
        assert cagrs == sorted(cagrs, reverse=True)

    def test_sort_by_rtr_first(self):
        """sort_by=(rtr, cagr) should rank by rtr descending."""
        matrix, returns, cash, configs = self._make_data()
        results, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999,
            sort_by=("rtr", "cagr")
        )
        rtrs = [r[0] for r in results]
        assert rtrs == sorted(rtrs, reverse=True)

    def test_different_sort_gives_different_order(self):
        """Different sort_by should produce different rankings."""
        matrix, returns, cash, configs = self._make_data()
        r_stab, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999,
            sort_by=("stability", "cagr")
        )
        r_rtr, _, _ = grid_search(
            matrix, returns, cash, configs, 10, 12.0, -999, -999,
            sort_by=("rtr", "cagr")
        )
        combos_stab = [r[4] for r in r_stab]
        combos_rtr = [r[4] for r in r_rtr]
        assert combos_stab != combos_rtr


class TestEvalAllCached:
    """Tests for _eval_all cached_metrics path."""

    def test_consistency_with_grid_search(self):
        """_eval_all output matches grid_search with cached_metrics."""
        np.random.seed(123)
        matrix = np.random.rand(24, 3).astype(np.float64)
        returns = np.random.normal(0.01, 0.04, 24).astype(np.float64)
        cash = np.full(24, 0.003, dtype=np.float64)
        configs = make_configs(3, 2, 2)
        metrics = _eval_all(matrix, returns, cash, configs, 12.0)
        results1, _, _ = grid_search(
            matrix, returns, cash, configs, 5, 12.0, -999, -999
        )
        results2, _, _ = grid_search(
            matrix, returns, cash, configs, 5, 12.0, -999, -999,
            cached_metrics=metrics
        )
        # Same results whether cached or freshly computed
        assert len(results1) == len(results2)
        for r1, r2 in zip(results1, results2):
            assert r1[4] == r2[4]  # same combo
            assert pytest.approx(r1[1]) == r2[1]  # same cagr
