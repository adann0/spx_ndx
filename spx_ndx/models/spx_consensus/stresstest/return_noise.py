"""Return noise injection - add noise to market returns, measure strategy robustness."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe, compute_strategy_returns,
)
from ._common import StressData


def return_noise_injection(data: StressData, rng, n_trials):
    """Add noise to market returns, measure strategy robustness."""
    n_periods = len(data.signal)
    returns_volatility = float(np.std(data.buy_hold_returns))
    noise_mults = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    results = {}

    for mult in noise_mults:
        noise_std = returns_volatility * mult
        trial_cagrs = np.empty(n_trials)
        trial_rtrs = np.empty(n_trials)
        trial_sharpes = np.empty(n_trials)

        for t in range(n_trials):
            noisy_bh = data.buy_hold_returns + rng.normal(0, noise_std, n_periods)
            strategy_returns = compute_strategy_returns(noisy_bh, data.signal, data.cash_returns)
            trial_cagrs[t] = compute_cagr(strategy_returns, data.periods_per_year)
            trial_rtrs[t] = compute_rtr(strategy_returns, data.periods_per_year)
            trial_sharpes[t] = compute_sharpe(strategy_returns, data.cash_returns, data.periods_per_year)

        results[mult] = {
            "cagrs": trial_cagrs.tolist(), "rtrs": trial_rtrs.tolist(),
            "sharpes": trial_sharpes.tolist(),
        }

    return {
        "noise_mults": noise_mults, "n_trials": n_trials,
        "returns_volatility": returns_volatility,
        "results": {str(k): v for k, v in results.items()},
    }
