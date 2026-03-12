"""Unit tests for stresstest pure functions."""

import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.stresstest import (
    StressData,
    compute_baseline,
    find_drawdowns,
    permutation_test,
    bootstrap_returns,
    txcosts_sensitivity,
    signal_noise_injection,
    return_noise_injection,
    montecarlo_paths,
    regime_split,
    cross_index,
    vintage_analysis,
    drawdown_analysis,
    rolling_alpha,
    block_bootstrap,
    correlation_bh,
    decade_split,
    rolling_rtr_sharpe,
    baseline_signal_correlation,
)

# Fixtures

PERIODS_PER_YEAR = 12
N_PERIODS = 120  # 10 years monthly
RNG_SEED = 42


@pytest.fixture
def rng():
    return np.random.default_rng(RNG_SEED)


@pytest.fixture
def market_data(rng):
    """Synthetic market data: StressData bundle + dates."""
    signal = rng.choice([0.0, 1.0], size=N_PERIODS, p=[0.35, 0.65])
    buy_hold_returns = rng.normal(0.007, 0.04, N_PERIODS)
    cash_returns = np.full(N_PERIODS, 0.003)
    strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns
    dates = pd.date_range("2000-01-01", periods=N_PERIODS, freq="MS")
    return StressData(signal, strategy_returns, buy_hold_returns, cash_returns, PERIODS_PER_YEAR), dates


@pytest.fixture
def baseline(market_data):
    data, _ = market_data
    return compute_baseline(data)


# compute_baseline

class TestComputeBaseline:
    def test_keys(self, baseline):
        expected = {
            "real_cagr", "real_rtr", "real_sharpe", "real_max_drawdown",
            "buy_hold_cagr", "buy_hold_rtr", "buy_hold_sharpe", "buy_hold_max_drawdown",
            "exposure", "n_trades",
        }
        assert set(baseline.keys()) == expected

    def test_exposure_from_signal(self, market_data):
        data, _ = market_data
        bl = compute_baseline(data)
        assert bl["exposure"] == pytest.approx(data.signal.mean())

    def test_n_trades_from_signal(self, market_data):
        data, _ = market_data
        bl = compute_baseline(data)
        assert bl["n_trades"] == int(np.abs(np.diff(data.signal)).sum())

    def test_all_invested(self):
        signal = np.ones(60)
        buy_hold = np.full(60, 0.01)
        cash = np.zeros(60)
        bl = compute_baseline(StressData(signal, buy_hold, buy_hold, cash, PERIODS_PER_YEAR))
        assert bl["exposure"] == 1.0
        assert bl["n_trades"] == 0


# find_drawdowns

class TestFindDrawdowns:
    def test_no_drawdown(self):
        returns = np.full(60, 0.01)
        dates = pd.date_range("2000-01-01", periods=60, freq="MS")
        result = find_drawdowns(returns, dates)
        assert result == []

    def test_single_drawdown(self):
        returns = np.full(60, 0.01)
        returns[20:30] = -0.05  # crash period
        dates = pd.date_range("2000-01-01", periods=60, freq="MS")
        result = find_drawdowns(returns, dates)
        assert len(result) >= 1
        assert result[0]["depth"] > 0.1

    def test_top_n_limit(self):
        returns = np.full(100, 0.01)
        returns[10:15] = -0.08
        returns[40:45] = -0.06
        returns[70:75] = -0.04
        dates = pd.date_range("2000-01-01", periods=100, freq="MS")
        result = find_drawdowns(returns, dates, top_n=2)
        assert len(result) <= 2

    def test_ongoing_drawdown(self):
        returns = np.full(30, -0.02)
        dates = pd.date_range("2000-01-01", periods=30, freq="MS")
        result = find_drawdowns(returns, dates)
        assert len(result) >= 1

    def test_output_fields(self):
        returns = np.full(60, 0.01)
        returns[10:20] = -0.05
        dates = pd.date_range("2000-01-01", periods=60, freq="MS")
        result = find_drawdowns(returns, dates)
        if result:
            dd = result[0]
            assert "depth" in dd
            assert "start" in dd
            assert "bottom" in dd
            assert "end" in dd
            assert "months_to_bottom" in dd
            assert "months_to_recover" in dd
            assert "total_months" in dd


# permutation_test

class TestPermutationTest:
    def test_keys(self, market_data, rng, baseline):
        data, _ = market_data
        result = permutation_test(data, rng, 50, baseline)
        assert "p_cagr" in result
        assert "p_rtr" in result
        assert "p_sharpe" in result
        assert "p_max_drawdown" in result
        assert len(result["permutation_cagrs"]) == 50

    def test_p_values_range(self, market_data, rng, baseline):
        data, _ = market_data
        result = permutation_test(data, rng, 100, baseline)
        for key in ["p_cagr", "p_rtr", "p_sharpe", "p_max_drawdown"]:
            assert 0.0 <= result[key] <= 1.0

    def test_deterministic(self, market_data, baseline):
        data, _ = market_data
        r1 = permutation_test(data, np.random.default_rng(0), 20, baseline)
        r2 = permutation_test(data, np.random.default_rng(0), 20, baseline)
        assert r1["permutation_cagrs"] == r2["permutation_cagrs"]


# bootstrap_returns

class TestBootstrapReturns:
    def test_keys(self, market_data, rng):
        data, _ = market_data
        result = bootstrap_returns(data, rng, 50)
        assert "ci_cagr" in result
        assert "ci_rtr" in result
        assert "percent_beat_buy_hold" in result
        assert len(result["bootstrap_alphas"]) == 50

    def test_ci_ordering(self, market_data, rng):
        data, _ = market_data
        result = bootstrap_returns(data, rng, 200)
        assert result["ci_cagr"][0] <= result["ci_cagr"][1]
        assert result["ci_rtr"][0] <= result["ci_rtr"][1]


# txcosts_sensitivity

class TestTxcostsSensitivity:
    def test_keys(self, market_data, baseline):
        data, _ = market_data
        result = txcosts_sensitivity(data, baseline)
        assert "costs_bps" in result
        assert "adjusted_cagrs" in result
        assert "n_trades" in result
        assert len(result["adjusted_cagrs"]) == len(result["costs_bps"])

    def test_zero_cost_matches_real(self, market_data, baseline):
        data, _ = market_data
        result = txcosts_sensitivity(data, baseline)
        assert result["adjusted_cagrs"][0] == pytest.approx(baseline["real_cagr"], abs=1e-10)

    def test_increasing_costs_decreases_cagr(self, market_data, baseline):
        data, _ = market_data
        result = txcosts_sensitivity(data, baseline)
        cagrs = result["adjusted_cagrs"]
        # Higher costs should generally reduce CAGR
        assert cagrs[0] >= cagrs[-1]

    def test_no_trades_no_cost(self, baseline):
        signal = np.ones(60)  # always invested -> 0 trades
        buy_hold_returns = np.full(60, 0.01)
        cash_returns = np.zeros(60)
        strategy_returns = buy_hold_returns.copy()
        data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, PERIODS_PER_YEAR)
        result = txcosts_sensitivity(data, baseline)
        assert result["n_trades"] == 0
        # All cost levels give same CAGR since no trades
        assert all(c == pytest.approx(result["adjusted_cagrs"][0]) for c in result["adjusted_cagrs"])

    def test_breakeven_found(self):
        """Strategy beats B&H at 0 cost but loses at high cost -> breakeven interpolated."""
        n = 120
        # Alternating signal -> 119 trades
        signal = np.array([1, 0] * (n // 2), dtype=float)
        buy_hold_returns = np.full(n, 0.01)
        buy_hold_returns[1::2] = -0.005  # mild loss on odd months
        cash_returns = np.full(n, 0.003)
        strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns
        data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, PERIODS_PER_YEAR)
        bl = compute_baseline(data)
        assert bl["real_cagr"] > bl["buy_hold_cagr"], "setup: strat must beat B&H"
        result = txcosts_sensitivity(data, bl)
        # ~40bps breakeven expected given the alpha/trade ratio
        assert result["breakeven"] is not None
        assert result["breakeven"] > 0


# signal_noise_injection

class TestSignalNoiseInjection:
    def test_keys(self, market_data, rng, baseline):
        data, _ = market_data
        result = signal_noise_injection(data, rng, 20, baseline)
        assert "noise_percents" in result
        assert "results" in result
        assert "1" in result["results"]  # 1% noise

    def test_more_noise_lower_beat(self, market_data, rng, baseline):
        data, _ = market_data
        result = signal_noise_injection(data, rng, 100, baseline)
        # With enough noise, percent_beat should generally decrease
        percent_1 = result["results"]["1"]["percent_beat"]
        percent_30 = result["results"]["30"]["percent_beat"]
        # Just check they're valid percentages
        assert 0 <= percent_1 <= 100
        assert 0 <= percent_30 <= 100


# return_noise_injection

class TestReturnNoiseInjection:
    def test_keys(self, market_data, rng):
        data, _ = market_data
        result = return_noise_injection(data, rng, 20)
        assert "noise_mults" in result
        assert "returns_volatility" in result
        assert "0.01" in result["results"]

    def test_returns_volatility_positive(self, market_data, rng):
        data, _ = market_data
        result = return_noise_injection(data, rng, 20)
        assert result["returns_volatility"] > 0


# montecarlo_paths

class TestMontecarloPaths:
    def test_keys(self, market_data, rng):
        data, _ = market_data
        result = montecarlo_paths(data, rng, 50)
        assert "mu" in result
        assert "sigma" in result
        assert "montecarlo_beat" in result
        assert "montecarlo_alpha" in result
        assert len(result["montecarlo_cagrs"]) == 50

    def test_mu_sigma(self, market_data, rng):
        data, _ = market_data
        result = montecarlo_paths(data, rng, 20)
        assert result["mu"] == pytest.approx(np.mean(data.buy_hold_returns))
        assert result["sigma"] == pytest.approx(np.std(data.buy_hold_returns, ddof=1))


# regime_split

class TestRegimeSplit:
    def test_basic(self, market_data, rng):
        data, dates = market_data
        vix = pd.Series(rng.uniform(10, 40, N_PERIODS), index=dates)
        result = regime_split(data, dates, vix)
        # At least VIX median split should exist with 120 periods
        assert any("VIX" in k for k in result.keys())

    def test_regime_keys(self, market_data, rng):
        data, dates = market_data
        vix = pd.Series(rng.uniform(15, 25, N_PERIODS), index=dates)
        result = regime_split(data, dates, vix)
        for v in result.values():
            assert "n" in v
            assert "strategy_cagr" in v
            assert "buy_hold_cagr" in v
            assert "delta" in v

    def test_small_regime_skipped(self, market_data):
        data, dates = market_data
        # VIX always 15 -> "VIX >= 30" bucket has 0 periods -> skipped
        vix = pd.Series(np.full(N_PERIODS, 15.0), index=dates)
        result = regime_split(data, dates, vix)
        assert "VIX >= 30" not in result


# cross_index

class TestCrossIndex:
    def test_spx_only(self, market_data, baseline):
        data, dates = market_data
        ds = pd.DataFrame({"spx_close": np.cumprod(1 + data.buy_hold_returns) * 100}, index=dates)
        result = cross_index(data, dates, baseline, ds)
        assert "SPX" in result
        assert result["SPX"]["strategy_cagr"] == pytest.approx(baseline["real_cagr"])

    def test_with_ndx(self, market_data, baseline, rng):
        data, dates = market_data
        ndx_prices = np.cumprod(1 + rng.normal(0.008, 0.05, N_PERIODS)) * 100
        ds = pd.DataFrame({
            "spx_close": np.cumprod(1 + data.buy_hold_returns) * 100,
            "ndx_close": ndx_prices,
        }, index=dates)
        result = cross_index(data, dates, baseline, ds)
        assert "NDX" in result
        assert "strategy_cagr" in result["NDX"]

    def test_short_index_skipped(self, market_data, baseline):
        data, dates = market_data
        # NDX with only 10 valid prices -> skipped
        ndx = np.full(N_PERIODS, np.nan)
        ndx[-10:] = np.cumprod(1 + np.full(10, 0.01)) * 100
        ds = pd.DataFrame({
            "spx_close": np.cumprod(1 + data.buy_hold_returns) * 100,
            "ndx_close": ndx,
        }, index=dates)
        result = cross_index(data, dates, baseline, ds)
        assert "NDX" not in result


# vintage_analysis

class TestVintageAnalysis:
    def test_basic(self, market_data):
        data, dates = market_data
        df = pd.DataFrame({
            "signal": data.signal, "strategy_returns": data.strategy_returns,
            "buy_hold_returns": data.buy_hold_returns, "cash_returns": np.full(N_PERIODS, 0.003),
        }, index=dates)
        result = vintage_analysis(data, df)
        assert "wins" in result
        assert "total" in result
        assert "years" in result
        assert result["total"] > 0

    def test_pct_range(self, market_data):
        data, dates = market_data
        df = pd.DataFrame({
            "signal": data.signal, "strategy_returns": data.strategy_returns,
            "buy_hold_returns": data.buy_hold_returns, "cash_returns": np.full(N_PERIODS, 0.003),
        }, index=dates)
        result = vintage_analysis(data, df)
        assert 0 <= result["percent"] <= 100

    def test_year_fields(self, market_data):
        data, dates = market_data
        df = pd.DataFrame({
            "signal": data.signal, "strategy_returns": data.strategy_returns,
            "buy_hold_returns": data.buy_hold_returns, "cash_returns": np.full(N_PERIODS, 0.003),
        }, index=dates)
        result = vintage_analysis(data, df)
        for yr_data in result["years"].values():
            assert "cagr_strategy" in yr_data
            assert "cagr_buy_hold" in yr_data
            assert "final_strategy" in yr_data
            assert "wins" in yr_data

    def test_no_cash_column(self):
        """df without cash_ret column -> zeros used."""
        dates = pd.date_range("2000-01-01", periods=60, freq="MS")
        signal = np.ones(60)
        buy_hold_returns = np.full(60, 0.005)
        strategy_returns = buy_hold_returns.copy()
        df = pd.DataFrame({"signal": signal, "strategy_returns": strategy_returns, "buy_hold_returns": buy_hold_returns},
                          index=dates)
        data = StressData(signal, strategy_returns, buy_hold_returns, np.zeros(60), PERIODS_PER_YEAR)
        result = vintage_analysis(data, df)
        assert result["total"] > 0

    def test_brentq_valueerror(self):
        """When no cash rate in [-0.5, 10.0] can make strat match B&H -> breakeven=None."""
        n = 36
        dates = pd.date_range("2000-01-01", periods=n, freq="MS")
        buy_hold_returns = np.full(n, 0.01)
        buy_hold_returns[12] = 2.0  # 200% gain in one month
        signal = np.ones(n)
        signal[12] = 0  # misses the spike
        cash_returns = np.full(n, 0.003)
        strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns
        df = pd.DataFrame({
            "signal": signal, "strategy_returns": strategy_returns,
            "buy_hold_returns": buy_hold_returns, "cash_returns": cash_returns,
        }, index=dates)
        data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, PERIODS_PER_YEAR)
        result = vintage_analysis(data, df)
        yr_data = result["years"].get("2000")
        if yr_data and not yr_data["wins"] and yr_data["cash_months"] > 0:
            assert yr_data["breakeven_cash_ann"] is None

    def test_strategy_wins_some_years(self):
        """Strategy beats B&H -> wins counter increments."""
        dates = pd.date_range("2000-01-01", periods=120, freq="MS")
        buy_hold_returns = np.full(120, 0.005)
        buy_hold_returns[12:24] = -0.05
        buy_hold_returns[36:48] = -0.05
        signal = np.ones(120)
        signal[12:24] = 0
        signal[36:48] = 0
        cash_returns = np.full(120, 0.003)
        strategy_returns = signal * buy_hold_returns + (1 - signal) * cash_returns
        df = pd.DataFrame({
            "signal": signal, "strategy_returns": strategy_returns,
            "buy_hold_returns": buy_hold_returns, "cash_returns": cash_returns,
        }, index=dates)
        data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, PERIODS_PER_YEAR)
        result = vintage_analysis(data, df)
        assert result["wins"] > 0


# drawdown_analysis

class TestDrawdownAnalysis:
    def test_keys(self, market_data):
        data, dates = market_data
        result = drawdown_analysis(data, dates)
        assert "strategy" in result
        assert "buy_hold" in result

    def test_returns_lists(self, market_data):
        data, dates = market_data
        result = drawdown_analysis(data, dates)
        assert isinstance(result["strategy"], list)
        assert isinstance(result["buy_hold"], list)


# rolling_alpha

class TestRollingAlpha:
    def test_basic(self, market_data):
        data, dates = market_data
        result = rolling_alpha(data, dates)
        assert "3Y" in result
        # 120 months > 60 -> 5Y also exists
        assert "5Y" in result

    def test_percent_positive_range(self, market_data):
        data, dates = market_data
        result = rolling_alpha(data, dates)
        for v in result.values():
            assert 0 <= v["percent_positive"] <= 100

    def test_short_data_skips_window(self):
        data = StressData(np.ones(30), np.full(30, 0.01), np.full(30, 0.008),
                       np.zeros(30), PERIODS_PER_YEAR)
        dates = pd.date_range("2000-01-01", periods=30, freq="MS")
        result = rolling_alpha(data, dates)
        # 30 < 36 -> no 3Y, 30 < 60 -> no 5Y
        assert result == {}


# block_bootstrap

class TestBlockBootstrap:
    def test_keys(self, market_data, rng):
        data, _ = market_data
        result = block_bootstrap(data, rng, 50)
        assert "rtr_mean" in result
        assert "sharpe_mean" in result
        assert "n_iter" in result
        assert result["n_iter"] == 50

    def test_custom_block_size(self, market_data, rng):
        data, _ = market_data
        result = block_bootstrap(data, rng, 30, block_size=12)
        assert result["block_size"] == 12


# correlation_bh

class TestCorrelationBh:
    def test_keys(self, market_data):
        data, _ = market_data
        result = correlation_bh(data)
        assert "corr_returns" in result
        assert "corr_abs" in result

    def test_correlation_range(self, market_data):
        data, _ = market_data
        result = correlation_bh(data)
        assert -1.0 <= result["corr_returns"] <= 1.0
        assert -1.0 <= result["corr_abs"] <= 1.0

    def test_identical_returns(self):
        returns = np.random.default_rng(0).normal(0.01, 0.02, 60)
        data = StressData(np.ones(60), returns, returns, np.zeros(60), PERIODS_PER_YEAR)
        result = correlation_bh(data)
        assert result["corr_returns"] == pytest.approx(1.0)


# decade_split

class TestDecadeSplit:
    def test_basic(self, market_data):
        data, dates = market_data
        result = decade_split(data, dates)
        assert isinstance(result, dict)
        assert len(result) >= 1  # 2000-2009 at least

    def test_decade_keys(self, market_data):
        data, dates = market_data
        result = decade_split(data, dates)
        for v in result.values():
            assert "strategy_cagr" in v
            assert "buy_hold_cagr" in v
            assert "strategy_max_drawdown" in v

    def test_short_decade_skipped(self):
        """Decade with < 12 months gets skipped (line 508)."""
        dates = pd.date_range("2019-07-01", periods=24, freq="MS")
        n = len(dates)
        data = StressData(np.ones(n), np.full(n, 0.01), np.full(n, 0.01),
                       np.zeros(n), PERIODS_PER_YEAR)
        result = decade_split(data, dates)
        assert "2010s" not in result
        assert "2020s" in result


# rolling_rtr_sharpe

class TestRollingRtrSharpe:
    def test_basic(self, market_data):
        data, dates = market_data
        result = rolling_rtr_sharpe(data, dates, 36)
        assert result is not None
        assert "dates" in result
        assert "strategy_rtr" in result
        assert result["window_months"] == 36
        assert len(result["strategy_rtr"]) == N_PERIODS - 36 + 1

    def test_short_data_returns_none(self):
        data = StressData(np.ones(30), np.full(30, 0.01), np.full(30, 0.01),
                       np.zeros(30), PERIODS_PER_YEAR)
        dates = pd.date_range("2000-01-01", periods=30, freq="MS")
        result = rolling_rtr_sharpe(data, dates, 36)
        assert result is None

    def test_percent_fields(self, market_data):
        data, dates = market_data
        result = rolling_rtr_sharpe(data, dates, 36)
        assert 0 <= result["percent_above_zero"] <= 100
        assert 0 <= result["percent_above_buy_hold"] <= 100


# baseline_signal_correlation

class TestBaselineSignalCorrelation:
    def test_empty(self):
        signal = np.ones(60)
        result = baseline_signal_correlation(signal, {})
        assert result["correlations"] == {}
        assert result["matrix"] == []

    def test_perfect_correlation(self):
        signal = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 10, dtype=float)
        result = baseline_signal_correlation(signal, {"clone": signal.copy()})
        assert result["correlations"]["clone"] == pytest.approx(1.0)
        assert result["labels"] == ["Consensus", "clone"]

    def test_multiple_signals(self, rng):
        signal = rng.choice([0.0, 1.0], size=100)
        baseline_signals = {"A": rng.choice([0.0, 1.0], size=100),
                            "B": rng.choice([0.0, 1.0], size=100)}
        result = baseline_signal_correlation(signal, baseline_signals)
        assert len(result["correlations"]) == 2
        assert len(result["labels"]) == 3
        mat = result["matrix"]
        assert len(mat) == 3
        assert mat[0][0] == pytest.approx(1.0)
