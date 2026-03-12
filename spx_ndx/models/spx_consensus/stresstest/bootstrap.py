"""Bootstrap returns - resample with replacement, compute confidence intervals."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe, compute_max_drawdown,
)
from ._common import StressData


def bootstrap_returns(data: StressData, rng, n_iter):
    """Resample returns with replacement, compute confidence intervals."""
    n = len(data.strategy_returns)
    bootstrap_cagrs = np.empty(n_iter)
    bootstrap_rtrs = np.empty(n_iter)
    bootstrap_sharpes = np.empty(n_iter)
    bootstrap_max_drawdowns = np.empty(n_iter)
    bootstrap_buy_hold_cagrs = np.empty(n_iter)

    for i in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        strategy_returns = data.strategy_returns[idx]
        buy_hold_returns = data.buy_hold_returns[idx]
        cash_returns = data.cash_returns[idx]
        bootstrap_cagrs[i] = compute_cagr(strategy_returns, data.periods_per_year)
        bootstrap_rtrs[i] = compute_rtr(strategy_returns, data.periods_per_year)
        bootstrap_sharpes[i] = compute_sharpe(strategy_returns, cash_returns, data.periods_per_year)
        bootstrap_max_drawdowns[i] = compute_max_drawdown(strategy_returns)
        bootstrap_buy_hold_cagrs[i] = compute_cagr(buy_hold_returns, data.periods_per_year)

    ci_cagr = np.percentile(bootstrap_cagrs, [2.5, 97.5]).tolist()
    ci_rtr = np.percentile(bootstrap_rtrs, [2.5, 97.5]).tolist()
    ci_sharpe = np.percentile(bootstrap_sharpes, [2.5, 97.5]).tolist()
    confidence_interval_max_drawdown = np.percentile(bootstrap_max_drawdowns, [2.5, 97.5]).tolist()

    return {
        "bootstrap_alphas": (bootstrap_cagrs - bootstrap_buy_hold_cagrs).tolist(),
        "bootstrap_cagrs": bootstrap_cagrs.tolist(),
        "ci_cagr": ci_cagr, "ci_rtr": ci_rtr,
        "ci_sharpe": ci_sharpe, "confidence_interval_max_drawdown": confidence_interval_max_drawdown,
        "percent_beat_buy_hold": float(np.mean(bootstrap_cagrs > bootstrap_buy_hold_cagrs) * 100),
    }
