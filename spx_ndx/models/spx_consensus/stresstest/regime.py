"""Regime split - evaluate strategy by VIX and market regimes."""

import numpy as np
import pandas as pd

from spx_ndx.models.spx_consensus.evaluate import (
    compute_cagr, compute_rtr, compute_sharpe, compute_strategy_returns,
)
from ._common import StressData


def regime_split(data: StressData, dates, vix):
    """Evaluate strategy by VIX and market regimes."""
    regimes = {}
    vix_median = vix.median()
    regimes["VIX < median"] = vix < vix_median
    regimes["VIX >= median"] = vix >= vix_median
    regimes["VIX < 20"] = vix < 20
    regimes["VIX 20-30"] = (vix >= 20) & (vix < 30)
    regimes["VIX >= 30"] = vix >= 30

    buy_hold_12_month_return = pd.Series(1 + data.buy_hold_returns, index=dates).rolling(12).apply(np.prod, raw=True) - 1
    regimes["Bull (12M > 0)"] = buy_hold_12_month_return > 0
    regimes["Bear (12M <= 0)"] = buy_hold_12_month_return <= 0

    cumulative = np.cumprod(1 + data.buy_hold_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown_series = pd.Series((peak - cumulative) / peak, index=dates)
    regimes["No DD (< 5%)"] = drawdown_series < 0.05
    regimes["Mild DD (5-15%)"] = (drawdown_series >= 0.05) & (drawdown_series < 0.15)
    regimes["Crisis (DD >= 15%)"] = drawdown_series >= 0.15

    result = {}
    for regime_name, regime_mask in regimes.items():
        regime_mask_values = regime_mask.values if hasattr(regime_mask, "values") else regime_mask
        n_regime_periods = int(regime_mask_values.sum())
        if n_regime_periods < 12:
            continue
        strategy_returns_regime = data.strategy_returns[regime_mask_values]
        buy_hold_returns_regime = data.buy_hold_returns[regime_mask_values]
        cash_returns_regime = data.cash_returns[regime_mask_values]
        regime_cagr_strategy = compute_cagr(strategy_returns_regime, data.periods_per_year)
        regime_cagr_buy_hold = compute_cagr(buy_hold_returns_regime, data.periods_per_year)
        result[regime_name] = {
            "n": n_regime_periods, "strategy_cagr": float(regime_cagr_strategy), "buy_hold_cagr": float(regime_cagr_buy_hold),
            "delta": float(regime_cagr_strategy - regime_cagr_buy_hold), "exposure": float(data.signal[regime_mask_values].mean()),
            "strategy_rtr": float(compute_rtr(strategy_returns_regime, data.periods_per_year)),
            "strategy_sharpe": float(compute_sharpe(strategy_returns_regime, cash_returns_regime, data.periods_per_year)),
        }
    return result
