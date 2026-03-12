"""Monte Carlo paths - apply signal to synthetic GBM paths, measure alpha."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_sharpe, compute_strategy_returns,
)
from ._common import StressData


def montecarlo_paths(data: StressData, rng, n_iter):
    """Apply signal to synthetic GBM paths, measure alpha."""
    n_periods = len(data.signal)
    mu = float(np.mean(data.buy_hold_returns))
    sigma = float(np.std(data.buy_hold_returns, ddof=1))

    montecarlo_cagrs = np.empty(n_iter)
    montecarlo_sharpes = np.empty(n_iter)
    montecarlo_buy_hold_cagrs = np.empty(n_iter)

    for i in range(n_iter):
        synthetic_returns = rng.normal(mu, sigma, n_periods)
        strategy_returns = compute_strategy_returns(synthetic_returns, data.signal, data.cash_returns)
        montecarlo_cagrs[i] = compute_cagr(strategy_returns, data.periods_per_year)
        montecarlo_sharpes[i] = compute_sharpe(strategy_returns, data.cash_returns, data.periods_per_year)
        montecarlo_buy_hold_cagrs[i] = compute_cagr(synthetic_returns, data.periods_per_year)

    return {
        "mu": mu, "sigma": sigma,
        "montecarlo_cagrs": montecarlo_cagrs.tolist(), "montecarlo_sharpes": montecarlo_sharpes.tolist(),
        "montecarlo_buy_hold_cagrs": montecarlo_buy_hold_cagrs.tolist(),
        "montecarlo_beat": float(np.mean(montecarlo_cagrs > montecarlo_buy_hold_cagrs) * 100),
        "montecarlo_alpha": float(np.mean(montecarlo_cagrs - montecarlo_buy_hold_cagrs)),
        "montecarlo_ci": np.percentile(montecarlo_cagrs, [2.5, 97.5]).tolist(),
    }
