"""Decade split - split performance by decade."""

import pandas as pd

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe, compute_max_drawdown,
)
from ._common import StressData


def decade_split(data: StressData, dates):
    """Split performance by decade."""
    years = pd.Series(dates).dt.year
    decades = {}
    for start in range(int(years.min() // 10 * 10), int(years.max()), 10):
        end = start + 9
        label = f"{start}s"
        mask = (years.values >= start) & (years.values <= end)
        if mask.sum() < 12:
            continue
        strategy_returns = data.strategy_returns[mask]
        buy_hold_returns = data.buy_hold_returns[mask]
        cash_returns = data.cash_returns[mask]
        decades[label] = {
            "strategy_cagr": float(compute_cagr(strategy_returns, data.periods_per_year)),
            "buy_hold_cagr": float(compute_cagr(buy_hold_returns, data.periods_per_year)),
            "strategy_rtr": float(compute_rtr(strategy_returns, data.periods_per_year)),
            "buy_hold_rtr": float(compute_rtr(buy_hold_returns, data.periods_per_year)),
            "strategy_sharpe": float(compute_sharpe(strategy_returns, cash_returns, data.periods_per_year)),
            "buy_hold_sharpe": float(compute_sharpe(buy_hold_returns, cash_returns, data.periods_per_year)),
            "strategy_max_drawdown": float(compute_max_drawdown(strategy_returns)),
            "buy_hold_max_drawdown": float(compute_max_drawdown(buy_hold_returns)),
        }
    return decades
