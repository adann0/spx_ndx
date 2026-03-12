"""Rolling RTR and Sharpe - over a fixed window."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import compute_rtr, compute_sharpe
from ._common import StressData


def rolling_rtr_sharpe(data: StressData, dates, window_months=36):
    """Rolling RTR and Sharpe over a fixed window."""
    n = len(data.strategy_returns)
    if n <= window_months:
        return None
    w = window_months
    strategy_rtrs, buy_hold_rtrs = [], []
    strategy_sharpes, buy_hold_sharpes = [], []
    rolling_dates = []

    for i in range(w, n + 1):
        window_strategy = data.strategy_returns[i - w:i]
        window_buy_hold = data.buy_hold_returns[i - w:i]
        window_cash = data.cash_returns[i - w:i]
        strategy_rtrs.append(compute_rtr(window_strategy, data.periods_per_year))
        buy_hold_rtrs.append(compute_rtr(window_buy_hold, data.periods_per_year))
        strategy_sharpes.append(compute_sharpe(window_strategy, window_cash, data.periods_per_year))
        buy_hold_sharpes.append(compute_sharpe(window_buy_hold, window_cash, data.periods_per_year))
        rolling_dates.append(str(dates[i - 1].date()))

    strategy_rtr_values = np.array(strategy_rtrs)
    buy_hold_rtr_values = np.array(buy_hold_rtrs)
    strategy_sharpe_values = np.array(strategy_sharpes)
    buy_hold_sharpe_values = np.array(buy_hold_sharpes)

    return {
        "dates": rolling_dates,
        "strategy_rtr": strategy_rtr_values.tolist(), "buy_hold_rtr": buy_hold_rtr_values.tolist(),
        "strategy_sharpe": strategy_sharpe_values.tolist(), "buy_hold_sharpe": buy_hold_sharpe_values.tolist(),
        "window_months": window_months,
        "mean": float(strategy_rtr_values.mean()), "min": float(strategy_rtr_values.min()), "max": float(strategy_rtr_values.max()),
        "percent_above_zero": float((strategy_rtr_values > 0).mean() * 100),
        "percent_above_buy_hold": float((strategy_rtr_values > buy_hold_rtr_values).mean() * 100),
        "sharpe_mean": float(strategy_sharpe_values.mean()),
        "sharpe_min": float(strategy_sharpe_values.min()), "sharpe_max": float(strategy_sharpe_values.max()),
        "percent_sharpe_above_zero": float((strategy_sharpe_values > 0).mean() * 100),
        "percent_sharpe_above_buy_hold": float((strategy_sharpe_values > buy_hold_sharpe_values).mean() * 100),
    }
