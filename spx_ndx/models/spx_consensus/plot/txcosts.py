"""Plot: Transaction costs sensitivity."""

import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, annotate_ax, footnote, BG, TEXT, GREEN, RED, ORANGE


def plot_txcosts(metrics, df, label):
    """CAGR vs transaction cost per trade bar chart."""
    base = metrics["baseline"]
    txcosts_metrics = metrics["txcosts"]
    bh_cagr = base["buy_hold_cagr"]
    costs_bps = txcosts_metrics["costs_bps"]
    adjusted_cagrs = txcosts_metrics["adjusted_cagrs"]
    n_trades = txcosts_metrics["n_trades"]
    breakeven = txcosts_metrics["breakeven"]
    days = (df.index[-1] - df.index[0]).days

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    cagr_vals = [c * 100 for c in adjusted_cagrs]
    colors = [GREEN if c >= bh_cagr else RED for c in adjusted_cagrs]
    ax.bar(range(len(costs_bps)), cagr_vals, color=colors, alpha=0.8,
           edgecolor=BG, width=0.7)
    ax.axhline(bh_cagr * 100, color=ORANGE, ls="--", lw=2,
               label="B&H")

    for i, (bps, cagr) in enumerate(zip(costs_bps, cagr_vals)):
        ax.text(i, cagr + 0.1, f"{cagr:.1f}%",
                ha="center", fontsize=9, fontweight="bold", color=TEXT)

    ax.set_xticks(range(len(costs_bps)))
    ax.set_xticklabels([f"{b/100:.2f}%" if b > 0 else "0" for b in costs_bps])
    ax.set_xlabel("Cost per trade (% of amount)", fontsize=11, color=TEXT)
    ax.set_ylabel("CAGR %", fontsize=11, color=TEXT)
    fig.suptitle("Transaction Costs",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    footnote(fig, f"{n_trades} trades over {days // 365} years")

    if breakeven:
        annotate_ax(ax, f"Breakeven ~{breakeven/100:.2f}%/trade")
    add_legend(ax, loc="upper right")

    path = "output/spx_consensus_stress_5_txcosts.png"
    save_fig(fig, path)
    return fig, path
