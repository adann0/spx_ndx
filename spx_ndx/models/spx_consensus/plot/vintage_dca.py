"""Plot: Vintage DCA small multiples and summary table."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._style import (
    apply_style, add_legend, save_fig, annotate_ax, style_table,
    TEXT, DIM, BG, GREEN, RED, TEAL,
)


def _irr(cashflows):
    """Compute monthly IRR via brentq, return annualized."""
    from scipy.optimize import brentq
    cashflows_array = np.array(cashflows, dtype=np.float64)
    t = np.arange(len(cashflows_array), dtype=np.float64)

    def npv(r):
        return np.sum(cashflows_array / (1 + r) ** t)
    try:
        monthly = brentq(npv, -0.3, 0.5, maxiter=500)
        return (1 + monthly) ** 12 - 1
    except (ValueError, RuntimeError):
        return np.nan


def _dca_equity(returns, n, initial, monthly):
    """Build equity curve with DCA: initial + monthly contribution."""
    equity = np.empty(n)
    equity[0] = initial * (1 + returns[0]) + monthly
    for t in range(1, n):
        equity[t] = equity[t-1] * (1 + returns[t]) + monthly
    return equity


def _compute_dca_vintages(start_years, df, signal, buy_hold_returns, tbill_monthly,
                          dca_initial, dca_monthly):
    """Compute DCA vintage metrics for each start year.

    Returns (dca_vintage, dca_wins, dca_total,
             irr_strategy, irr_buy_hold, max_drawdowns_strategy, max_drawdowns_buy_hold).
    """
    dca_wins, dca_total = 0, 0
    dca_vintage = {}
    irr_strategy, irr_buy_hold, max_drawdowns_strategy, max_drawdowns_buy_hold = [], [], [], []

    for yr in start_years:
        mask = df.index.year >= yr
        n = int(mask.sum())
        sub_signal = signal[mask]
        sub_buy_hold = buy_hold_returns[mask]
        sub_tbill = tbill_monthly[mask]
        strategy_monthly_returns = sub_signal * sub_buy_hold + (1 - sub_signal) * sub_tbill

        equity_strategy = _dca_equity(strategy_monthly_returns, n, dca_initial, dca_monthly)
        equity_buy_hold = _dca_equity(sub_buy_hold, n, dca_initial, dca_monthly)
        cashflows_strategy = [-(dca_initial + dca_monthly)] + [-dca_monthly] * (n - 1) + [equity_strategy[-1]]
        cashflows_buy_hold = [-(dca_initial + dca_monthly)] + [-dca_monthly] * (n - 1) + [equity_buy_hold[-1]]

        irr_strategy_value = _irr(cashflows_strategy)
        irr_buy_hold_value = _irr(cashflows_buy_hold)
        irr_strategy.append(irr_strategy_value)
        irr_buy_hold.append(irr_buy_hold_value)
        max_drawdowns_strategy.append(float(((equity_strategy - np.maximum.accumulate(equity_strategy))
                                / np.maximum.accumulate(equity_strategy)).min()))
        max_drawdowns_buy_hold.append(float(((equity_buy_hold - np.maximum.accumulate(equity_buy_hold))
                                / np.maximum.accumulate(equity_buy_hold)).min()))

        strategy_wins = bool(equity_strategy[-1] >= equity_buy_hold[-1])
        if strategy_wins:
            dca_wins += 1
        dca_total += 1

        cash_months = int((sub_signal == 0).sum())
        avg_tbill_ann = (
            float((1 + np.mean(sub_tbill[sub_signal == 0]))**12 - 1)
            if cash_months > 0 else 0.0
        )
        breakeven = None
        if not strategy_wins and cash_months > 0:
            def _breakeven_objective(r_ann, _sig=sub_signal, _bh=sub_buy_hold, _n=n):
                r_m = (1 + r_ann) ** (1/12) - 1
                enhanced = _sig * _bh + (1 - _sig) * r_m
                eq_e = _dca_equity(enhanced, _n, dca_initial, dca_monthly)
                eq_b2 = _dca_equity(_bh, _n, dca_initial, dca_monthly)
                return eq_e[-1] - eq_b2[-1]
            try:
                from scipy.optimize import brentq as _brentq
                breakeven = float(_brentq(_breakeven_objective, -0.5, 10.0, xtol=1e-6))
            except (ValueError, ImportError):
                pass

        dca_vintage[str(yr)] = {
            "equity_strategy": equity_strategy, "equity_buy_hold": equity_buy_hold,
            "final_strategy": float(equity_strategy[-1]),
            "final_buy_hold": float(equity_buy_hold[-1]),
            "irr_strategy": float(irr_strategy_value) if not np.isnan(irr_strategy_value) else None,
            "irr_buy_hold": float(irr_buy_hold_value) if not np.isnan(irr_buy_hold_value) else None,
            "wins": strategy_wins,
            "cash_months": cash_months,
            "avg_tbill_ann": avg_tbill_ann,
            "breakeven_cash_ann": breakeven,
        }

    return dca_vintage, dca_wins, dca_total, irr_strategy, irr_buy_hold, max_drawdowns_strategy, max_drawdowns_buy_hold


def plot_vintage_dca(metrics, df, label, signal, buy_hold_returns, cash_returns,
                     dca_initial=300_000, dca_monthly=3_000, dataset=None):
    """DCA vintage small multiples; returns list of paths."""
    vintage_metrics = metrics["vintage"]
    start_years = sorted([int(y) for y in vintage_metrics["years"].keys()])

    if dataset is None:
        dataset_dca = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset_dca.index = pd.to_datetime(dataset_dca.index)
    else:
        dataset_dca = dataset
    tbill_monthly = (
        ((1 + dataset_dca["tbill_rate"] / 100) ** (1/12) - 1)
        .reindex(df.index).fillna(0).values
    )

    dca_vintage, dca_wins, dca_total, _, _, _, _ = _compute_dca_vintages(
        start_years, df, signal, buy_hold_returns, tbill_monthly, dca_initial, dca_monthly)

    paths = []
    spread_label = f"{dca_initial//1000}K+{dca_monthly//1000}K/mo"
    file_suffix = "dca"

    n_columns = 5
    n_rows_dca = int(np.ceil(len(start_years) / n_columns))
    fig, axes = plt.subplots(n_rows_dca, n_columns,
                             figsize=(4.2 * n_columns, 3.2 * n_rows_dca))
    axes = np.atleast_2d(axes)
    apply_style(fig, axes.flatten())

    for i, yr in enumerate(start_years):
        ax = axes[i // n_columns, i % n_columns]
        year_data = dca_vintage[str(yr)]
        equity_strategy = year_data["equity_strategy"]
        equity_buy_hold = year_data["equity_buy_hold"]
        strategy_wins = year_data["wins"]
        irr_strategy_value = year_data["irr_strategy"]
        irr_buy_hold_value = year_data["irr_buy_hold"]
        mask = df.index.year >= yr
        sub = df[mask]

        ax.set_facecolor("#0f1d0f" if strategy_wins else "#1d0f0f")
        x = sub.index
        ax.plot(x, equity_buy_hold / 1000, color=RED, lw=1.2, alpha=0.7,
                label="B&H")
        ax.plot(x, equity_strategy / 1000, color=TEAL, lw=1.8, label="Strat")
        ax.fill_between(x, equity_buy_hold / 1000, equity_strategy / 1000,
                        where=equity_strategy >= equity_buy_hold, color=GREEN, alpha=0.15,
                        interpolate=True)
        ax.fill_between(x, equity_buy_hold / 1000, equity_strategy / 1000,
                        where=equity_strategy < equity_buy_hold, color=RED, alpha=0.15,
                        interpolate=True)

        final_strategy = equity_strategy[-1] / 1000
        final_buy_hold = equity_buy_hold[-1] / 1000
        ax.set_title(f"Start {yr}", fontsize=9, fontweight="bold",
                     color=TEXT)
        diff_percent = (equity_strategy[-1] / equity_buy_hold[-1] - 1) * 100
        sign = "+" if diff_percent >= 0 else ""
        irr_strategy_string = (f"{irr_strategy_value:+.1%}"
                     if irr_strategy_value is not None and not np.isnan(irr_strategy_value)
                     else "N/A")
        irr_buy_hold_string = (f"{irr_buy_hold_value:+.1%}"
                     if irr_buy_hold_value is not None and not np.isnan(irr_buy_hold_value)
                     else "N/A")
        ax.annotate(
            f"Strat: {final_strategy:,.0f}K (IRR {irr_strategy_string})\n"
            f"B&H:  {final_buy_hold:,.0f}K (IRR {irr_buy_hold_string})\n"
            f"{sign}{diff_percent:.0f}%",
            xy=(0.03, 0.97), xycoords="axes fraction",
            fontsize=6.5, va="top", ha="left", linespacing=1.4,
            fontweight="bold", color=GREEN if strategy_wins else RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.85,
                      edgecolor="none"))

        ax.set_ylabel("K$", fontsize=7, color=DIM)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", rotation=45)

    for j in range(len(start_years), n_rows_dca * n_columns):
        axes[j // n_columns, j % n_columns].set_visible(False)

    dca_percent = dca_wins / dca_total * 100
    fig.suptitle("Vintage Year (DCA)",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    plt.tight_layout()
    path = f"output/spx_consensus_stress_6_vintage_{file_suffix}.png"
    save_fig(fig, path)
    paths.append(path)

    return paths


