"""Rolling alpha - compute rolling CAGR alpha over 3Y and 5Y windows."""

import pandas as pd

from spx_ndx.models.spx_consensus.evaluate import compute_cagr
from ._common import StressData


def rolling_alpha(data: StressData, dates):
    """Compute rolling CAGR alpha over 3Y and 5Y windows."""
    strategy_series = pd.Series(data.strategy_returns, index=dates)
    buy_hold_series = pd.Series(data.buy_hold_returns, index=dates)
    windows = {"3Y": 3 * data.periods_per_year, "5Y": 5 * data.periods_per_year}
    result = {}

    for window_name, w in windows.items():
        if len(data.strategy_returns) <= w:
            continue
        rolling_strategy = strategy_series.rolling(w).apply(lambda x: compute_cagr(x.values, data.periods_per_year), raw=False)
        rolling_buy_hold = buy_hold_series.rolling(w).apply(lambda x: compute_cagr(x.values, data.periods_per_year), raw=False)
        alpha = (rolling_strategy - rolling_buy_hold).dropna()
        result[window_name] = {
            "mean": float(alpha.mean()), "min": float(alpha.min()),
            "max": float(alpha.max()),
            "percent_positive": float((alpha > 0).mean() * 100),
        }
    return result
