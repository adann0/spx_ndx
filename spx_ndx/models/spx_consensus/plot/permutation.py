"""Plot: Permutation test histogram."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, stat_line, footnote, BG, DIM, RED, TEXT


def plot_permutation(metrics, df, label):
    """Histogram of permuted CAGRs vs real strategy CAGR."""
    meta = metrics["meta"]
    base = metrics["baseline"]
    permutation_metrics = metrics["permutation"]
    real_cagr = base["real_cagr"]
    permutation_cagrs = np.array(permutation_metrics["permutation_cagrs"])
    p_cagr = permutation_metrics["p_cagr"]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    ax.hist(permutation_cagrs * 100, bins=50, color=DIM, alpha=0.8, edgecolor=BG, lw=0.5)
    ax.axvline(real_cagr * 100, color=RED, lw=2, zorder=5,
               label="Strategy")
    ax.axvline(permutation_cagrs.mean() * 100, color=DIM, lw=1.5, ls="--",
               label="Random mean")

    pctile = (1 - p_cagr) * 100
    stat_line(ax, f"p={p_cagr:.3f}  percentile={pctile:.1f}%  "
                  f"strategy={real_cagr*100:.1f}%  random_mean={permutation_cagrs.mean()*100:.1f}%")

    ax.set_xlabel("CAGR %", fontsize=12, color=TEXT)
    ax.set_ylabel("Number of permutations", fontsize=11, color=TEXT)
    add_legend(ax)
    fig.suptitle("Permutation Test",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    footnote(fig, f"{meta['n_iter']} shuffles")

    path = "output/spx_consensus_stress_1_permutation.png"
    save_fig(fig, path)
    return fig, path
