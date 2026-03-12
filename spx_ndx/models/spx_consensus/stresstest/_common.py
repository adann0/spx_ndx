"""Shared types and helpers for stress test functions."""

from typing import NamedTuple

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe,
    compute_max_drawdown,
)


class StressData(NamedTuple):
    """Common arrays shared by all stress tests."""
    signal: np.ndarray
    strategy_returns: np.ndarray
    buy_hold_returns: np.ndarray
    cash_returns: np.ndarray
    periods_per_year: int


def compute_baseline(data: StressData):
    """Compute baseline metrics for strategy and B&H."""
    return {
        "real_cagr": compute_cagr(data.strategy_returns, data.periods_per_year),
        "real_rtr": compute_rtr(data.strategy_returns, data.periods_per_year),
        "real_sharpe": compute_sharpe(data.strategy_returns, data.cash_returns, data.periods_per_year),
        "real_max_drawdown": compute_max_drawdown(data.strategy_returns),
        "buy_hold_cagr": compute_cagr(data.buy_hold_returns, data.periods_per_year),
        "buy_hold_rtr": compute_rtr(data.buy_hold_returns, data.periods_per_year),
        "buy_hold_sharpe": compute_sharpe(data.buy_hold_returns, data.cash_returns, data.periods_per_year),
        "buy_hold_max_drawdown": compute_max_drawdown(data.buy_hold_returns),
        "exposure": float(data.signal.mean()),
        "n_trades": int(np.abs(np.diff(data.signal)).sum()),
    }


def find_drawdowns(returns, dates, top_n=5):
    """Find top N drawdown episodes from periodic returns."""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    episodes = []
    in_drawdown = False
    start_index = 0
    for i in range(len(drawdown)):
        if drawdown[i] > 0.01 and not in_drawdown:
            in_drawdown = True
            start_index = i
        elif drawdown[i] < 0.001 and in_drawdown:
            in_drawdown = False
            bottom_index = start_index + np.argmax(drawdown[start_index:i + 1])
            episodes.append((drawdown[bottom_index], start_index, bottom_index, i))
    if in_drawdown:
        bottom_index = start_index + np.argmax(drawdown[start_index:])
        episodes.append((drawdown[bottom_index], start_index, bottom_index, len(drawdown) - 1))
    episodes.sort(key=lambda x: -x[0])
    result = []
    for depth, start_index, bottom_index, end_index in episodes[:top_n]:
        result.append({
            "depth": float(depth),
            "start": str(dates[start_index].date()), "bottom": str(dates[bottom_index].date()),
            "end": str(dates[end_index].date()),
            "months_to_bottom": bottom_index - start_index,
            "months_to_recover": end_index - bottom_index,
            "total_months": end_index - start_index,
        })
    return result
