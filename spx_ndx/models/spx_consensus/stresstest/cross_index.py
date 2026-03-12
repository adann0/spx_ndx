"""Cross-index test - apply SPX signal to other indices."""

import numpy as np

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe,
    compute_max_drawdown, compute_strategy_returns,
)
from ._common import StressData


def cross_index(data: StressData, dates, baseline, dataset):
    """Apply SPX signal to other indices."""
    result = {
        "SPX": {
            "strategy_cagr": baseline["real_cagr"], "strategy_rtr": baseline["real_rtr"],
            "strategy_sharpe": baseline["real_sharpe"], "strategy_max_drawdown": baseline["real_max_drawdown"],
            "buy_hold_cagr": baseline["buy_hold_cagr"], "buy_hold_rtr": baseline["buy_hold_rtr"],
            "buy_hold_sharpe": baseline["buy_hold_sharpe"], "buy_hold_max_drawdown": baseline["buy_hold_max_drawdown"],
            "exposure": float(data.signal.mean()), "n": len(data.signal),
        }
    }
    for index_name, col in {"NDX": "ndx_close", "MSCI World": "msci_close"}.items():
        if col not in dataset.columns:
            continue
        index_prices = dataset[col].reindex(dates).ffill()
        index_returns = index_prices.pct_change()
        valid = index_returns.notna()
        if valid.sum() < 24:
            continue
        index_returns_values = index_returns[valid].values
        signal_values = data.signal[valid.values]
        cash_returns_values = data.cash_returns[valid.values]
        strategy_returns = compute_strategy_returns(index_returns_values, signal_values, cash_returns_values)
        result[index_name] = {
            "strategy_cagr": float(compute_cagr(strategy_returns, data.periods_per_year)),
            "strategy_rtr": float(compute_rtr(strategy_returns, data.periods_per_year)),
            "strategy_sharpe": float(compute_sharpe(strategy_returns, cash_returns_values, data.periods_per_year)),
            "strategy_max_drawdown": float(compute_max_drawdown(strategy_returns)),
            "buy_hold_cagr": float(compute_cagr(index_returns_values, data.periods_per_year)),
            "buy_hold_rtr": float(compute_rtr(index_returns_values, data.periods_per_year)),
            "buy_hold_sharpe": float(compute_sharpe(index_returns_values, cash_returns_values, data.periods_per_year)),
            "buy_hold_max_drawdown": float(compute_max_drawdown(index_returns_values)),
            "exposure": float(signal_values.mean()), "n": int(valid.sum()),
        }
    return result
