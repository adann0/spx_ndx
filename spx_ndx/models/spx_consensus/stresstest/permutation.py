"""Permutation test - shuffle signal timing, measure how often random beats real."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe,
    compute_max_drawdown, compute_strategy_returns,
)
from ._common import StressData


def permutation_test(data: StressData, rng, n_iter, baseline):
    """Shuffle signal timing, measure how often random beats real."""
    n = len(data.signal)
    permutation_cagrs = np.empty(n_iter)
    permutation_rtrs = np.empty(n_iter)
    permutation_sharpes = np.empty(n_iter)
    permutation_max_drawdowns = np.empty(n_iter)

    for i in range(n_iter):
        shuf = data.signal.copy()
        rng.shuffle(shuf)
        strategy_returns = compute_strategy_returns(data.buy_hold_returns, shuf, data.cash_returns)
        permutation_cagrs[i] = compute_cagr(strategy_returns, data.periods_per_year)
        permutation_rtrs[i] = compute_rtr(strategy_returns, data.periods_per_year)
        permutation_sharpes[i] = compute_sharpe(strategy_returns, data.cash_returns, data.periods_per_year)
        permutation_max_drawdowns[i] = compute_max_drawdown(strategy_returns)

    return {
        "permutation_cagrs": permutation_cagrs.tolist(),
        "permutation_rtrs": permutation_rtrs.tolist(),
        "permutation_sharpes": permutation_sharpes.tolist(),
        "permutation_max_drawdowns": permutation_max_drawdowns.tolist(),
        "p_cagr": float(np.mean(permutation_cagrs >= baseline["real_cagr"])),
        "p_rtr": float(np.mean(permutation_rtrs >= baseline["real_rtr"])),
        "p_sharpe": float(np.mean(permutation_sharpes >= baseline["real_sharpe"])),
        "p_max_drawdown": float(np.mean(permutation_max_drawdowns <= baseline["real_max_drawdown"])),
    }
