"""Plot: Drawdown depth vs duration scatter."""

import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, stat_line, TEXT, BLUE, RED


def plot_drawdown_scatter(metrics, df, label):
    """Drawdown depth vs duration scatter for strategy and B&H."""
    drawdown_metrics = metrics["drawdowns"]

    fig, ax = plt.subplots(figsize=(9, 6))
    apply_style(fig, ax)
    if drawdown_metrics["strategy"]:
        strategy_x = [d["total_months"] for d in drawdown_metrics["strategy"]]
        strategy_y = [d["depth"] * 100 for d in drawdown_metrics["strategy"]]
        ax.scatter(strategy_x, strategy_y, c=BLUE, s=80, zorder=3, label="Strategy",
                   edgecolors=TEXT, lw=1.5)
        for i, d in enumerate(drawdown_metrics["strategy"]):
            ax.annotate(f"#{i+1}", (d["total_months"], d["depth"] * 100),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=BLUE)
    if drawdown_metrics["buy_hold"]:
        buy_hold_x = [d["total_months"] for d in drawdown_metrics["buy_hold"]]
        buy_hold_y = [d["depth"] * 100 for d in drawdown_metrics["buy_hold"]]
        ax.scatter(buy_hold_x, buy_hold_y, c=RED, s=80, zorder=3, label="B&H",
                   edgecolors=TEXT, lw=1.5)
        for i, d in enumerate(drawdown_metrics["buy_hold"]):
            ax.annotate(f"#{i+1}", (d["total_months"], d["depth"] * 100),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=RED)
    ax.set_xlabel("Total duration (months)", fontsize=11, color=TEXT)
    ax.set_ylabel("Depth %", fontsize=11, color=TEXT)
    fig.suptitle("Drawdowns", fontsize=13,
                 fontweight="bold", color=TEXT, y=1.01)
    add_legend(ax)

    worst_strategy = drawdown_metrics["strategy"][0] if drawdown_metrics["strategy"] else None
    worst_buy_hold = drawdown_metrics["buy_hold"][0] if drawdown_metrics["buy_hold"] else None
    if worst_strategy and worst_buy_hold:
        stat_line(ax, f"worst_DD_strat={worst_strategy['depth']:.1%} ({worst_strategy['total_months']}m)  "
                      f"worst_DD_B&H={worst_buy_hold['depth']:.1%} ({worst_buy_hold['total_months']}m)")

    path = "output/spx_consensus_stress_9_scatter.png"
    save_fig(fig, path)
    return fig, path
