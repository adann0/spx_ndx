"""Plot: Decade split bar charts."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, TEXT, DIM, GREEN, ORANGE


def _decade_bars(axes, labels, strategy_cagr, buy_hold_cagr, strategy_mid, buy_hold_mid, strategy_max_drawdown, buy_hold_max_drawdown,
                 mid_label):
    """Fill 3 axes with CAGR / mid-metric / MaxDD decade bars."""
    x = np.arange(len(labels))
    bar_width = 0.35

    for ax, strategy_values, buy_hold_values, ylabel, title, format_value, offset in [
        (axes[0], strategy_cagr, buy_hold_cagr, "CAGR %", "CAGR per decade",
         lambda v: f"{v:+.1f}%", 0.2),
        (axes[1], strategy_mid, buy_hold_mid, mid_label, f"{mid_label} per decade",
         lambda v: f"{v:.2f}", 0.02),
        (axes[2], strategy_max_drawdown, buy_hold_max_drawdown, "Max DD %", "Max Drawdown per decade",
         lambda v: f"{v:.1f}%", 0.3),
    ]:
        ax.bar(x - bar_width/2, strategy_values, bar_width, color=GREEN, alpha=0.8, label="Strategy")
        ax.bar(x + bar_width/2, buy_hold_values, bar_width, color=DIM, alpha=0.8, label="B&H")
        for i in range(len(labels)):
            ax.text(i - bar_width/2, strategy_values[i] + offset, format_value(strategy_values[i]),
                    ha="center", fontsize=9, color=TEXT)
            ax.text(i + bar_width/2, buy_hold_values[i] + offset, format_value(buy_hold_values[i]),
                    ha="center", fontsize=9, color=DIM)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, color=TEXT, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT)
        add_legend(ax)


def plot_decades(metrics, df, label):
    """CAGR, Sharpe, and MaxDD bar charts per decade."""
    if "decades" not in metrics or not metrics["decades"]:
        return None, None

    decades_data = metrics["decades"]
    decade_labels = list(decades_data.keys())
    strategy_cagr = [decades_data[k]["strategy_cagr"] * 100 for k in decade_labels]
    buy_hold_cagr = [decades_data[k]["buy_hold_cagr"] * 100 for k in decade_labels]
    strategy_max_drawdown_values = [decades_data[k]["strategy_max_drawdown"] * 100 for k in decade_labels]
    buy_hold_max_drawdown_values = [decades_data[k]["buy_hold_max_drawdown"] * 100 for k in decade_labels]

    _has_sharpe = all("strategy_sharpe" in decades_data[k] and "buy_hold_sharpe" in decades_data[k]
                      for k in decade_labels)
    if not _has_sharpe:
        return None, None

    sharpe_strategy = [decades_data[k]["strategy_sharpe"] for k in decade_labels]
    sharpe_buy_hold = [decades_data[k]["buy_hold_sharpe"] for k in decade_labels]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    apply_style(fig, axes)
    _decade_bars(axes, decade_labels, strategy_cagr, buy_hold_cagr, sharpe_strategy, sharpe_buy_hold,
                 strategy_max_drawdown_values, buy_hold_max_drawdown_values, "Sharpe")
    fig.suptitle("Decade Split",
                 fontsize=14, fontweight="bold", color=TEXT, y=1.02)
    plt.tight_layout()
    path = "output/spx_consensus_stress_7_decades.png"
    save_fig(fig, path)
    return fig, path
