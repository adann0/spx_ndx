"""Drawdown analysis - find top drawdown episodes for strategy and B&H."""

from ._common import StressData, find_drawdowns


def drawdown_analysis(data: StressData, dates):
    """Find top drawdown episodes for strategy and B&H."""
    return {
        "strategy": find_drawdowns(data.strategy_returns, dates),
        "buy_hold": find_drawdowns(data.buy_hold_returns, dates),
    }
