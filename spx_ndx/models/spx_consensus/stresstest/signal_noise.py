"""Signal noise injection - flip random fractions of signal, measure degradation."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe, compute_strategy_returns,
)
from ._common import StressData


def signal_noise_injection(data: StressData, rng, n_trials, baseline):
    """Flip random fractions of signal, measure degradation."""
    n_periods = len(data.signal)
    noise_percents = [1, 2, 5, 10, 15, 20, 30]
    bh_cagr = baseline["buy_hold_cagr"]
    results = {}

    for percent in noise_percents:
        n_flip = max(1, int(n_periods * percent / 100))
        trial_cagrs = np.empty(n_trials)
        trial_rtrs = np.empty(n_trials)
        trial_sharpes = np.empty(n_trials)

        for t in range(n_trials):
            noisy = data.signal.copy()
            flip_idx = rng.choice(n_periods, size=n_flip, replace=False)
            noisy[flip_idx] = 1 - noisy[flip_idx]
            strategy_returns = compute_strategy_returns(data.buy_hold_returns, noisy, data.cash_returns)
            trial_cagrs[t] = compute_cagr(strategy_returns, data.periods_per_year)
            trial_rtrs[t] = compute_rtr(strategy_returns, data.periods_per_year)
            trial_sharpes[t] = compute_sharpe(strategy_returns, data.cash_returns, data.periods_per_year)

        results[percent] = {
            "cagrs": trial_cagrs.tolist(), "rtrs": trial_rtrs.tolist(),
            "sharpes": trial_sharpes.tolist(),
            "percent_beat": float(np.mean(trial_cagrs > bh_cagr) * 100),
        }

    return {
        "noise_percents": noise_percents, "n_trials": n_trials,
        "results": {str(k): v for k, v in results.items()},
    }
