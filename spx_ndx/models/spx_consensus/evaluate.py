"""Evaluation functions for strategy returns and performance metrics."""

import numpy as np


def compute_strategy_returns(market_returns, signal, cash_returns):
    """Compute strategy returns by blending market and cash based on signal."""
    return signal * market_returns + (1 - signal) * cash_returns


def compute_cagr(returns, periods_per_year=12):
    """Compute annualized compound growth rate from periodic returns."""
    cumulative = np.prod(1 + returns)
    n_years = len(returns) / periods_per_year
    return cumulative ** (1 / n_years) - 1


def compute_annual_volatility(returns, periods_per_year=12):
    """Compute annualized volatility from periodic returns."""
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)


def compute_rtr(returns, periods_per_year=12):
    """Annualized reward-to-risk ratio (mean/std, no risk-free subtraction).

    Used internally for ranking. NOT a Sharpe ratio.
    """
    std = np.std(returns, ddof=1)
    if std < 1e-12:
        return 0.0
    return (np.mean(returns) / std) * np.sqrt(periods_per_year)


def compute_sharpe(returns, cash_returns, periods_per_year=12):
    """Annualized Sharpe ratio (excess returns over risk-free)."""
    excess = returns - cash_returns
    std = np.std(excess, ddof=1)
    if std < 1e-12:
        return 0.0
    return (np.mean(excess) / std) * np.sqrt(periods_per_year)


def compute_max_drawdown(returns):
    """Maximum drawdown from periodic returns (positive float, e.g. 0.20 for 20%)."""
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (peak - cumulative) / peak
    return np.max(drawdown)


def compute_turnover(signal, periods_per_year=12):
    """Annualized turnover from a binary signal."""
    changes = np.sum(np.abs(np.diff(signal)))
    n_diffs = len(signal) - 1
    if n_diffs == 0:
        return 0.0
    return (changes / n_diffs) * periods_per_year


def compute_stability(returns):
    """Stability as R-squared of log cumulative returns vs time (0-1)."""
    log_cumulative = np.log(np.cumprod(1 + returns))
    x = np.arange(len(log_cumulative))
    x_mean = np.mean(x)
    y_mean = np.mean(log_cumulative)
    sum_cross = np.sum((x - x_mean) * (log_cumulative - y_mean))
    sum_squares_x = np.sum((x - x_mean) ** 2)
    sum_squares_y = np.sum((log_cumulative - y_mean) ** 2)
    if sum_squares_x == 0 or sum_squares_y == 0:
        return 1.0
    correlation = sum_cross / np.sqrt(sum_squares_x * sum_squares_y)
    return correlation ** 2


def compute_hit_rate(returns):
    """Fraction of positive return periods."""
    return np.mean(returns > 0)


def compute_all_metrics(returns, signal, periods_per_year=12, cash_returns=None):
    """Compute all performance metrics in one call.

    Returns dict with keys: cagr, volatility, rtr, max_drawdown, turnover,
    stability, hit_rate. If cash_returns provided, also includes sharpe.
    """
    result = {
        "cagr": compute_cagr(returns, periods_per_year),
        "volatility": compute_annual_volatility(returns, periods_per_year),
        "rtr": compute_rtr(returns, periods_per_year),
        "max_drawdown": compute_max_drawdown(returns),
        "turnover": compute_turnover(signal, periods_per_year),
        "stability": compute_stability(returns),
        "hit_rate": compute_hit_rate(returns),
    }
    if cash_returns is not None:
        result["sharpe"] = compute_sharpe(returns, cash_returns, periods_per_year)
    return result
