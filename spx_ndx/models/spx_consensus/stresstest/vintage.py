"""Vintage analysis - for each start year, check if strategy beats B&H."""

import numpy as np
from scipy.optimize import brentq

from spx_ndx.models.spx_consensus.evaluate import compute_cagr
from ._common import StressData


def vintage_analysis(data: StressData, df):
    """For each start year, check if strategy beats B&H."""
    start_years = sorted(set(df.index.year))
    start_years = [y for y in start_years if y <= df.index[-1].year - 1]

    tbill_col = df.get("cash_returns")
    if tbill_col is None:
        tbill_monthly = np.zeros(len(df))
    else:
        tbill_monthly = tbill_col.values

    wins = 0
    total = 0
    results = {}

    for yr in start_years:
        mask = df.index.year >= yr
        subset_signal = data.signal[mask]
        subset_buy_hold = data.buy_hold_returns[mask]
        subset_strategy = data.strategy_returns[mask]
        subset_tbill = tbill_monthly[mask]
        equity_strategy = np.cumprod(1 + subset_strategy)
        equity_buy_hold = np.cumprod(1 + subset_buy_hold)
        n_periods = int(mask.sum())
        n_days = (df.index[mask][-1] - df.index[mask][0]).days
        local_ppy = data.periods_per_year if n_days < 365 else round(n_periods / (n_days / 365.25))
        cagr_strategy = compute_cagr(subset_strategy, local_ppy)
        cagr_buy_hold = compute_cagr(subset_buy_hold, local_ppy)
        strategy_wins = bool(equity_strategy[-1] >= equity_buy_hold[-1])
        if strategy_wins:
            wins += 1
        total += 1

        cash_months = int((subset_signal == 0).sum())
        avg_tbill_ann = float((1 + np.mean(subset_tbill[subset_signal == 0])) ** 12 - 1) if cash_months > 0 else 0.0
        breakeven = None
        if not strategy_wins and cash_months > 0:
            def _obj(r_ann, _sig=subset_signal, _bh=subset_buy_hold):
                r_m = (1 + r_ann) ** (1 / 12) - 1
                enhanced = _sig * _bh + (1 - _sig) * r_m
                return np.prod(1 + enhanced) - np.prod(1 + _bh)
            try:
                breakeven = float(brentq(_obj, -0.5, 10.0, xtol=1e-6))
            except ValueError:
                breakeven = None

        results[str(yr)] = {
            "cagr_strategy": float(cagr_strategy), "cagr_buy_hold": float(cagr_buy_hold),
            "final_strategy": float(equity_strategy[-1] * 100_000),
            "final_buy_hold": float(equity_buy_hold[-1] * 100_000),
            "wins": strategy_wins,
            "cash_months": cash_months,
            "avg_tbill_ann": avg_tbill_ann,
            "breakeven_cash_ann": breakeven,
        }

    return {"wins": wins, "total": total, "percent": wins / total * 100 if total else 0, "years": results}
