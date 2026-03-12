import numpy as np
import pytest

from spx_ndx.models.spx_consensus.evaluate import (
    compute_strategy_returns,
    compute_cagr,
    compute_annual_volatility,
    compute_rtr,
    compute_sharpe,
    compute_max_drawdown,
    compute_turnover,
    compute_stability,
    compute_hit_rate,
    compute_all_metrics,
)


class TestComputeStrategyReturns:
    """Tests for compute_strategy_returns."""

    def test_fully_invested(self):
        """When signal is all 1s, strategy returns = market returns."""
        market = np.array([0.01, -0.02, 0.03])
        signal = np.array([1.0, 1.0, 1.0])
        cash = np.array([0.001, 0.001, 0.001])
        result = compute_strategy_returns(market, signal, cash)
        np.testing.assert_array_almost_equal(result, market)

    def test_fully_cash(self):
        """When signal is all 0s, strategy returns = cash returns."""
        market = np.array([0.01, -0.02, 0.03])
        signal = np.array([0.0, 0.0, 0.0])
        cash = np.array([0.001, 0.001, 0.001])
        result = compute_strategy_returns(market, signal, cash)
        np.testing.assert_array_almost_equal(result, cash)

    def test_mixed_signal(self):
        """Invested when signal=1, cash when signal=0."""
        market = np.array([0.01, -0.02, 0.03])
        signal = np.array([1.0, 0.0, 1.0])
        cash = np.array([0.001, 0.001, 0.001])
        result = compute_strategy_returns(market, signal, cash)
        expected = np.array([0.01, 0.001, 0.03])
        np.testing.assert_array_almost_equal(result, expected)

    def test_returns_new_array(self):
        """Must return a new array, not modify inputs."""
        market = np.array([0.01, -0.02])
        signal = np.array([1.0, 0.0])
        cash = np.array([0.001, 0.001])
        result = compute_strategy_returns(market, signal, cash)
        assert result is not market
        assert result is not cash


class TestComputeCAGR:
    """Tests for compute_cagr."""

    def test_zero_returns(self):
        """All zero returns -> CAGR = 0."""
        rets = np.zeros(12)
        assert compute_cagr(rets, periods_per_year=12) == pytest.approx(0.0)

    def test_constant_monthly_return(self):
        """1% monthly for 12 months -> CAGR ≈ 12.68%."""
        rets = np.full(12, 0.01)
        expected = (1.01 ** 12) ** (12 / 12) - 1  # ~0.1268
        assert compute_cagr(rets, periods_per_year=12) == pytest.approx(expected, rel=1e-6)

    def test_two_years(self):
        """24 months of 0.5% -> annualized correctly."""
        rets = np.full(24, 0.005)
        cum = (1.005 ** 24)
        years = 24 / 12
        expected = cum ** (1 / years) - 1
        assert compute_cagr(rets, periods_per_year=12) == pytest.approx(expected, rel=1e-6)


class TestComputeAnnualVol:
    """Tests for compute_annual_volatility."""

    def test_constant_returns(self):
        """Constant returns -> vol = 0."""
        rets = np.full(12, 0.01)
        assert compute_annual_volatility(rets, periods_per_year=12) == pytest.approx(0.0)

    def test_known_vol(self):
        """Known monthly std -> annualized by sqrt(12)."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=1200)
        monthly_std = np.std(rets, ddof=1)
        expected = monthly_std * np.sqrt(12)
        assert compute_annual_volatility(rets, periods_per_year=12) == pytest.approx(expected, rel=1e-6)


class TestComputeRTR:
    """Tests for compute_rtr."""

    def test_zero_vol(self):
        """Constant returns -> RTR = 0 (avoid division by zero)."""
        rets = np.full(12, 0.01)
        assert compute_rtr(rets, periods_per_year=12) == 0.0

    def test_positive_rtr(self):
        """Positive mean, some vol -> positive RTR."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=120)
        result = compute_rtr(rets, periods_per_year=12)
        assert result > 0

    def test_annualized(self):
        """RTR = mean/std * sqrt(periods_per_year)."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=120)
        expected = (np.mean(rets) / np.std(rets, ddof=1)) * np.sqrt(12)
        assert compute_rtr(rets, periods_per_year=12) == pytest.approx(expected, rel=1e-6)


class TestComputeSharpe:
    """Tests for compute_sharpe (excess returns over risk-free)."""

    def test_zero_excess(self):
        """Returns equal to cash -> Sharpe = 0."""
        rets = np.array([0.01, 0.02, 0.03])
        cash = np.array([0.01, 0.02, 0.03])
        assert compute_sharpe(rets, cash, periods_per_year=12) == 0.0

    def test_positive_sharpe(self):
        """Strategy consistently beats cash -> positive Sharpe."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=120)
        cash = np.full(120, 0.003)
        result = compute_sharpe(rets, cash, periods_per_year=12)
        assert result > 0

    def test_formula(self):
        """Sharpe = mean(excess) / std(excess) * sqrt(ppy)."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=120)
        cash = np.full(120, 0.003)
        excess = rets - cash
        expected = (np.mean(excess) / np.std(excess, ddof=1)) * np.sqrt(12)
        assert compute_sharpe(rets, cash, periods_per_year=12) == pytest.approx(expected, rel=1e-6)

    def test_lower_than_rtr(self):
        """Sharpe (excess) should be lower than RTR (total) when cash > 0."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.01, 0.04, size=120)
        cash = np.full(120, 0.003)
        assert compute_sharpe(rets, cash) < compute_rtr(rets)

    def test_zero_vol_excess(self):
        """Constant excess returns -> Sharpe = 0 (avoid division by zero)."""
        rets = np.full(12, 0.01)
        cash = np.full(12, 0.005)
        assert compute_sharpe(rets, cash, periods_per_year=12) == 0.0


class TestComputeMaxDrawdown:
    """Tests for compute_max_drawdown."""

    def test_always_up(self):
        """Monotonically increasing -> max DD = 0."""
        rets = np.array([0.01, 0.02, 0.03, 0.01])
        assert compute_max_drawdown(rets) == pytest.approx(0.0)

    def test_single_drop(self):
        """Simple drop and recovery."""
        # Start at 1, go to 1.1, drop to 1.1*0.9=0.99, recover to 0.99*1.05
        rets = np.array([0.10, -0.10, 0.05])
        # Peak at 1.1, trough at 0.99 -> DD = (1.1 - 0.99) / 1.1 = 0.1/1.1 ≈ 0.0909
        assert compute_max_drawdown(rets) == pytest.approx(0.10, abs=0.01)

    def test_returns_positive_value(self):
        """Max DD is returned as a positive number."""
        rets = np.array([0.05, -0.20, 0.10])
        assert compute_max_drawdown(rets) > 0


class TestComputeTurnover:
    """Tests for compute_turnover."""

    def test_no_changes(self):
        """Constant signal -> turnover = 0."""
        signal = np.array([1.0, 1.0, 1.0, 1.0])
        assert compute_turnover(signal, periods_per_year=12) == 0.0

    def test_alternating(self):
        """Signal flips every period -> high turnover."""
        signal = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        result = compute_turnover(signal, periods_per_year=12)
        assert result > 0

    def test_single_flip(self):
        """One flip in 12 periods -> turnover = 1/year."""
        signal = np.zeros(12)
        signal[6:] = 1.0  # one flip at index 6
        result = compute_turnover(signal, periods_per_year=12)
        # 1 change in 11 diffs, annualized: (1/11)*12 ≈ 1.09
        expected = (1 / 11) * 12
        assert result == pytest.approx(expected, rel=1e-6)

    def test_single_element(self):
        """Single-element signal -> no diffs -> turnover = 0."""
        signal = np.array([1.0])
        assert compute_turnover(signal, periods_per_year=12) == 0.0


class TestComputeStability:
    """Tests for compute_stability (R² of log cumulative returns)."""

    def test_perfect_trend(self):
        """Constant returns -> R² ≈ 1."""
        rets = np.full(60, 0.01)
        assert compute_stability(rets) == pytest.approx(1.0, abs=1e-6)

    def test_noisy_returns(self):
        """Random returns -> R² < 1."""
        rng = np.random.RandomState(42)
        rets = rng.normal(0.005, 0.05, size=120)
        result = compute_stability(rets)
        assert 0 <= result <= 1

    def test_single_return(self):
        """Single return -> degenerate (ss_xx=0) -> stability = 1."""
        rets = np.array([0.01])
        assert compute_stability(rets) == 1.0


class TestComputeHitRate:
    """Tests for compute_hit_rate."""

    def test_all_positive(self):
        """All positive returns -> 100%."""
        rets = np.array([0.01, 0.02, 0.03])
        assert compute_hit_rate(rets) == pytest.approx(1.0)

    def test_all_negative(self):
        """All negative returns -> 0%."""
        rets = np.array([-0.01, -0.02, -0.03])
        assert compute_hit_rate(rets) == pytest.approx(0.0)

    def test_mixed(self):
        """2 positive, 1 negative -> 66.7%."""
        rets = np.array([0.01, -0.02, 0.03])
        assert compute_hit_rate(rets) == pytest.approx(2 / 3, rel=1e-6)

    def test_zero_counts_as_non_positive(self):
        """Zero return is not positive."""
        rets = np.array([0.0, 0.01])
        assert compute_hit_rate(rets) == pytest.approx(0.5)


class TestComputeAllMetrics:
    """Tests for compute_all_metrics convenience function."""

    def test_returns_dict(self):
        """Returns a dict with all expected keys."""
        rets = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
        signal = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        result = compute_all_metrics(rets, signal)
        expected_keys = {
            "cagr", "volatility", "rtr",
            "max_drawdown", "turnover", "stability", "hit_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_values_are_floats(self):
        """All values must be plain floats."""
        rets = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
        signal = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        result = compute_all_metrics(rets, signal)
        for v in result.values():
            assert isinstance(v, (float, np.floating))

    def test_with_cash_returns(self):
        """When cash_returns provided, result includes sharpe key."""
        rets = np.array([0.01, -0.02, 0.03, 0.005, -0.01, 0.02])
        signal = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0])
        cash = np.full(6, 0.003)
        result = compute_all_metrics(rets, signal, cash_returns=cash)
        assert "sharpe" in result
        assert isinstance(result["sharpe"], (float, np.floating))
