"""Plot: Regime split bar chart."""

import matplotlib.pyplot as plt

from ._style import apply_style, save_fig, TEXT, DIM, BG, GREEN, RED


def plot_regimes(metrics, df, label):
    """Alpha per VIX/trend regime bar chart."""
    regime_metrics = metrics["regimes"]

    key_regimes = ["VIX < 20", "VIX 20-30", "VIX >= 30",
                   "Bull (12M > 0)", "Bear (12M <= 0)"]
    key_regimes = [r for r in key_regimes if r in regime_metrics]
    key_labels = {
        "VIX < 20": "Calm market\n(VIX < 20)",
        "VIX 20-30": "Nervous market\n(VIX 20-30)",
        "VIX >= 30": "Crisis / panic\n(VIX >= 30)",
        "Bull (12M > 0)": "Bull market\n(12M > 0)",
        "Bear (12M <= 0)": "Bear market\n(12M <= 0)",
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    apply_style(fig, ax)
    x = range(len(key_regimes))
    deltas = [regime_metrics[r]["delta"] * 100 for r in key_regimes]
    bar_colors = [GREEN if d > 0 else RED for d in deltas]
    ax.bar(x, deltas, color=bar_colors, alpha=0.8, edgecolor=BG, width=0.6)
    ax.axhline(0, color=DIM, lw=1)

    for i, (d, r) in enumerate(zip(deltas, key_regimes)):
        n_regime_periods = regime_metrics[r]["n"]
        expo = regime_metrics[r]["exposure"] * 100
        offset_val = 1.0 if d > 0 else -1.0
        offset_info = 2.5 if d > 0 else -2.5
        ax.text(i, d + offset_val, f"{d:+.1f} pp",
                ha="center", fontsize=11, fontweight="bold", color=bar_colors[i])
        ax.text(i, d + offset_info, f"({n_regime_periods} months, expo {expo:.0f}%)",
                ha="center", fontsize=8, color=DIM)

    ax.set_xticks(list(x))
    ax.set_xticklabels([key_labels.get(r, r) for r in key_regimes], fontsize=10)
    ax.set_ylabel("Alpha vs B&H (percentage points)", fontsize=11, color=TEXT)
    fig.suptitle("Regimes",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 3, ymax + 5)

    path = "output/spx_consensus_stress_8_regimes.png"
    save_fig(fig, path)
    return fig, path
