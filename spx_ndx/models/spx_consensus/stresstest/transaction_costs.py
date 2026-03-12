"""Transaction cost sensitivity - evaluate strategy under various cost levels."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe,
)
from ._common import StressData


def txcosts_sensitivity(data: StressData, baseline):
    """Evaluate strategy under various transaction cost levels."""
    costs_bps = [0, 5, 10, 15, 20, 30, 50, 75, 100]
    trades = np.abs(np.diff(data.signal))
    n_trades = int(trades.sum())
    n_periods = len(data.signal)

    adjusted_cagrs, adjusted_rtrs, adjusted_sharpes = [], [], []
    for bps in costs_bps:
        cost = bps / 10000
        adjusted_returns = data.strategy_returns.copy()
        for j in range(1, n_periods):
            if trades[j - 1] == 1:
                adjusted_returns[j] -= cost
        adjusted_cagrs.append(compute_cagr(adjusted_returns, data.periods_per_year))
        adjusted_rtrs.append(compute_rtr(adjusted_returns, data.periods_per_year))
        adjusted_sharpes.append(compute_sharpe(adjusted_returns, data.cash_returns, data.periods_per_year))

    breakeven = None
    bh_cagr = baseline["buy_hold_cagr"]
    for i in range(len(costs_bps) - 1):
        if adjusted_cagrs[i] >= bh_cagr and adjusted_cagrs[i + 1] < bh_cagr:
            fraction = (adjusted_cagrs[i] - bh_cagr) / (adjusted_cagrs[i] - adjusted_cagrs[i + 1])
            breakeven = costs_bps[i] + fraction * (costs_bps[i + 1] - costs_bps[i])
            break

    return {
        "costs_bps": costs_bps,
        "adjusted_cagrs": [float(c) for c in adjusted_cagrs],
        "adjusted_rtrs": [float(r) for r in adjusted_rtrs],
        "adjusted_sharpes": [float(s) for s in adjusted_sharpes],
        "n_trades": n_trades,
        "breakeven": float(breakeven) if breakeven else None,
    }
