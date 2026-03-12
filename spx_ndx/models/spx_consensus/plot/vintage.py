"""Plot: Vintage year small multiples."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, save_fig, TEXT, DIM, BG, GREEN, RED, TEAL


def plot_vintage(metrics, df, label):
    """Small multiples: $100K invested at each start year, strat vs B&H."""
    vintage_metrics = metrics["vintage"]
    start_years = sorted([int(y) for y in vintage_metrics["years"].keys()])

    n_columns = 5
    n_rows = int(np.ceil(len(start_years) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(4.2 * n_columns, 3.2 * n_rows))
    axes = np.atleast_2d(axes)
    apply_style(fig, axes.flatten())

    for i, yr in enumerate(start_years):
        ax = axes[i // n_columns, i % n_columns]
        sub = df[df.index.year >= yr]

        equity_strategy = 100_000 * np.cumprod(1 + sub["strategy_returns"].values)
        equity_buy_hold = 100_000 * np.cumprod(1 + sub["buy_hold_returns"].values)

        year_data = vintage_metrics["years"][str(yr)]
        strategy_wins = year_data["wins"]

        ax.set_facecolor("#0f1d0f" if strategy_wins else "#1d0f0f")
        x = sub.index
        ax.plot(x, equity_buy_hold / 1000, color=RED, lw=1.2, alpha=0.7, label="B&H")
        ax.plot(x, equity_strategy / 1000, color=TEAL, lw=1.8, label="Strat")
        ax.fill_between(x, equity_buy_hold / 1000, equity_strategy / 1000,
                        where=equity_strategy >= equity_buy_hold, color=GREEN, alpha=0.15,
                        interpolate=True)
        ax.fill_between(x, equity_buy_hold / 1000, equity_strategy / 1000,
                        where=equity_strategy < equity_buy_hold, color=RED, alpha=0.15,
                        interpolate=True)

        final_strategy = equity_strategy[-1] / 1000
        final_buy_hold = equity_buy_hold[-1] / 1000
        ax.set_title(f"Start {yr}", fontsize=9, fontweight="bold", color=TEXT)
        diff_percent = (equity_strategy[-1] / equity_buy_hold[-1] - 1) * 100
        sign = "+" if diff_percent >= 0 else ""
        ax.annotate(
            f"Strat: {final_strategy:,.0f}K ({year_data['cagr_strategy']:+.1%})\n"
            f"B&H:  {final_buy_hold:,.0f}K ({year_data['cagr_buy_hold']:+.1%})\n"
            f"{sign}{diff_percent:.0f}%",
            xy=(0.03, 0.97), xycoords="axes fraction",
            fontsize=6.5, va="top", ha="left", linespacing=1.4,
            fontweight="bold", color=GREEN if strategy_wins else RED,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=BG, alpha=0.85,
                      edgecolor="none"))

        ax.set_ylabel("K$", fontsize=7, color=DIM)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", rotation=45)

    for j in range(len(start_years), n_rows * n_columns):
        axes[j // n_columns, j % n_columns].set_visible(False)

    wins = vintage_metrics["wins"]
    total = vintage_metrics["total"]
    percent = vintage_metrics["percent"]
    fig.suptitle("Vintage Year (Lump Sum)",
        fontsize=14, fontweight="bold", color=TEXT, y=1.01)
    plt.tight_layout()

    path = "output/spx_consensus_stress_6_vintage.png"
    save_fig(fig, path)
    return fig, path
