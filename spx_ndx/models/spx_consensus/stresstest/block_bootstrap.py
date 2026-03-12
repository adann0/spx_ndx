"""Block bootstrap - preserving autocorrelation."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_rtr, compute_sharpe, compute_strategy_returns,
)
from ._common import StressData


def block_bootstrap(data: StressData, rng, n_iter, block_size=6):
    """Block bootstrap preserving autocorrelation."""
    n_periods = len(data.signal)
    n_blocks = int(np.ceil(n_periods / block_size))
    bootstrap_rtrs = np.empty(n_iter)
    bootstrap_sharpes = np.empty(n_iter)

    for i in range(n_iter):
        starts = rng.integers(0, n_periods - block_size + 1, size=n_blocks)
        synthetic_buy_hold = np.concatenate([data.buy_hold_returns[s:s + block_size] for s in starts])[:n_periods]
        synthetic_cash = np.concatenate([data.cash_returns[s:s + block_size] for s in starts])[:n_periods]
        strategy_returns = compute_strategy_returns(synthetic_buy_hold, data.signal, synthetic_cash)
        bootstrap_rtrs[i] = compute_rtr(strategy_returns, data.periods_per_year)
        bootstrap_sharpes[i] = compute_sharpe(strategy_returns, synthetic_cash, data.periods_per_year)

    return {
        "rtr_mean": float(np.mean(bootstrap_rtrs)),
        "rtr_std": float(np.std(bootstrap_rtrs)),
        "rtr_p5": float(np.percentile(bootstrap_rtrs, 5)),
        "rtr_p95": float(np.percentile(bootstrap_rtrs, 95)),
        "rtr_percent_positive": float((bootstrap_rtrs > 0).mean()),
        "sharpe_mean": float(np.mean(bootstrap_sharpes)),
        "sharpe_std": float(np.std(bootstrap_sharpes)),
        "sharpe_p5": float(np.percentile(bootstrap_sharpes, 5)),
        "sharpe_p95": float(np.percentile(bootstrap_sharpes, 95)),
        "n_iter": n_iter, "block_size": block_size,
    }
