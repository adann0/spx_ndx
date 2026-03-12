"""Plot: Cumulative alpha over time."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, save_fig, stat_line, TEXT, DIM, GREEN, RED


def plot_cumulative_alpha(metrics, df, label, strategy_returns, buy_hold_returns):
    """Cumulative alpha (strategy - B&H) over time."""
    fig, ax = plt.subplots(figsize=(14, 5))
    apply_style(fig, ax)
    cumulative_alpha = np.cumsum(strategy_returns - buy_hold_returns) * 100
    ax.fill_between(df.index, cumulative_alpha, 0, where=cumulative_alpha > 0,
                    color=GREEN, alpha=0.4)
    ax.fill_between(df.index, cumulative_alpha, 0, where=cumulative_alpha <= 0,
                    color=RED, alpha=0.4)
    ax.plot(df.index, cumulative_alpha, color=TEXT, lw=1.5)
    ax.axhline(0, color=DIM, lw=0.5)
    ax.set_ylabel("Cumulative alpha (pp)", fontsize=10, color=TEXT)
    fig.suptitle("Cumulative Alpha", fontsize=13,
                 fontweight="bold", color=TEXT, y=1.01)
    stat_line(ax, f"cumulative_alpha={cumulative_alpha[-1]:+.1f} pp  "
                  f"min={cumulative_alpha.min():+.1f}  max={cumulative_alpha.max():+.1f}")

    path = "output/spx_consensus_stress_11_cumalpha.png"
    save_fig(fig, path)
    return fig, path
