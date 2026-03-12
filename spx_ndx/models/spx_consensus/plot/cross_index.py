"""Plot: Cross-index equity/metrics/drawdown panels."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._style import (
    apply_style, add_legend, save_fig, stat_line, TEXT, DIM, BG, GRID, BORDER,
    BLUE, GREEN, RED, ORANGE,
)
from ..evaluate import compute_strategy_returns


def _cross_index_panel(dates, strategy_returns, buy_hold_returns, strategy_signal, index_name, r, series_color,
                       label, mid_metric, title_suffix, path_suffix):
    """Render one cross-index panel (equity + drawdown)."""
    middle_metric_name, middle_metric_strategy, middle_metric_buy_hold = mid_metric
    strategy_cumulative = 100 * np.cumprod(1 + strategy_returns)
    buy_hold_cumulative = 100 * np.cumprod(1 + buy_hold_returns)

    fig, (axis_equity, axis_drawdown) = plt.subplots(2, 1, figsize=(14, 7),
                                                      height_ratios=[2, 1])
    apply_style(fig, [axis_equity, axis_drawdown])

    axis_equity.plot(dates, buy_hold_cumulative, color=DIM, lw=1, alpha=0.7,
               label="B&H")
    axis_equity.plot(dates, strategy_cumulative, color=series_color, lw=1.5,
               label="Strategy")
    for i in range(1, len(dates)):
        color = GREEN if strategy_signal[i] == 1 else RED
        axis_equity.axvspan(dates[i-1], dates[i], color=color, alpha=0.05)
    stat_line(axis_equity, f"Strategy={r['strategy_cagr']:+.1%}  B&H={r['buy_hold_cagr']:+.1%}")
    axis_equity.set_ylabel("Equity (100)", color=TEXT)
    add_legend(axis_equity)

    for series_returns_loop, series_label, c, lw in [
        (strategy_returns, "Strategy", series_color, 1.5),
        (buy_hold_returns, "B&H", DIM, 1),
    ]:
        cumulative_returns = np.cumprod(1 + series_returns_loop)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown_values = -(peak - cumulative_returns) / peak * 100
        axis_drawdown.plot(dates, drawdown_values, color=c, lw=lw, alpha=0.8, label=series_label)
    cumulative_returns_strategy = np.cumprod(1 + strategy_returns)
    peak_strategy = np.maximum.accumulate(cumulative_returns_strategy)
    axis_drawdown.fill_between(dates, -(peak_strategy - cumulative_returns_strategy) / peak_strategy * 100, 0,
                       color=series_color, alpha=0.15)
    stat_line(axis_drawdown, f"Strategy={r['strategy_max_drawdown']:.1%}  B&H={r['buy_hold_max_drawdown']:.1%}")
    axis_drawdown.set_ylabel("Drawdown %", color=TEXT)
    axis_drawdown.set_title("Drawdowns", fontsize=10, fontweight="bold", color=TEXT)
    add_legend(axis_drawdown)

    fig.suptitle(f"{index_name}",
                 fontsize=13, fontweight="bold", color=TEXT)
    fig.subplots_adjust(hspace=0.35, top=0.92)
    safe_name = index_name.replace(" ", "_").lower()
    path = f"output/spx_consensus_cross_validation_{safe_name}.png"
    save_fig(fig, path)
    return path


def plot_cross_index(metrics, df, label, signal, strategy_returns, buy_hold_returns, cash_returns,
                     dataset=None):
    """Equity, metrics, and drawdown for SPX signal applied on NDX/MSCI World."""
    if dataset is None:
        frequency = metrics["meta"].get("frequency", "monthly")
        dataset = pd.read_parquet(f"datas/dataset_{frequency}.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    from ..stresstest import cross_index as _cross_index, StressData, compute_baseline
    periods_per_year = metrics["meta"].get("periods_per_year", 12)
    data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, periods_per_year)
    baseline = metrics.get("baseline") or compute_baseline(data)
    cross_index_data = _cross_index(data, df.index, baseline, dataset)

    index_colors = {"SPX": RED, "NDX": BLUE, "MSCI World": GREEN}
    cross_cols = {"NDX": "ndx_close", "MSCI World": "msci_close"}

    paths = []
    for index_name in ["SPX", "NDX", "MSCI World"]:
        if index_name not in cross_index_data:
            continue
        r = cross_index_data[index_name]

        if index_name == "SPX":
            strat_ret, bh_ret, strat_sig, dates = strategy_returns, buy_hold_returns, signal, df.index
        else:
            column_name = cross_cols[index_name]
            if column_name not in dataset.columns:
                continue
            index_prices = dataset[column_name].reindex(df.index).ffill()
            index_returns = index_prices.pct_change()
            valid = index_returns.notna()
            bh_ret = index_returns[valid].values
            strat_sig = signal[valid.values]
            strat_ret = compute_strategy_returns(bh_ret, strat_sig, cash_returns[valid.values])
            dates = df.index[valid.values]

        series_color = index_colors.get(index_name, BLUE)
        if "strategy_sharpe" in r and "buy_hold_sharpe" in r:
            paths.append(_cross_index_panel(
                dates, strat_ret, bh_ret, strat_sig, index_name, r, series_color, label,
                ("Sharpe", r["strategy_sharpe"], r["buy_hold_sharpe"]), "", ""))

    return paths
