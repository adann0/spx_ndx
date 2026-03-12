"""Plot: Bootstrap alpha distribution."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, stat_line, footnote, TEXT, DIM, BLUE, GREEN, RED


def plot_bootstrap(metrics, df, label):
    """KDE of bootstrapped alpha distribution vs zero."""
    from scipy.stats import gaussian_kde

    meta = metrics["meta"]
    base = metrics["baseline"]
    bootstrap_metrics = metrics["bootstrap"]
    bootstrap_alphas = np.array(bootstrap_metrics["bootstrap_alphas"]) * 100
    confidence_interval_alpha_low, confidence_interval_alpha_high = np.percentile(bootstrap_alphas, [2.5, 97.5])
    percent_beat_buy_hold = bootstrap_metrics["percent_beat_buy_hold"]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)

    kde = gaussian_kde(bootstrap_alphas, bw_method=0.3)
    x_kde = np.linspace(bootstrap_alphas.min() - 1, bootstrap_alphas.max() + 1, 500)
    y_kde = kde(x_kde)

    ax.plot(x_kde, y_kde, color=TEXT, lw=1.5, zorder=3)
    ax.fill_between(x_kde, y_kde, where=x_kde >= 0, color=GREEN, alpha=0.3, zorder=2)
    ax.fill_between(x_kde, y_kde, where=x_kde < 0, color=RED, alpha=0.3, zorder=2)
    ax.axvline(0, color=DIM, lw=1, ls="-", zorder=4)
    ax.axvline(np.mean(bootstrap_alphas), color=BLUE, lw=1.5, ls="--", zorder=4,
               label="Mean alpha")

    stat_line(ax, f"beats_B&H={percent_beat_buy_hold:.0f}%  "
                  f"95% CI=[{confidence_interval_alpha_low:+.1f}, {confidence_interval_alpha_high:+.1f}] pp  "
                  f"mean={np.mean(bootstrap_alphas):+.1f} pp")

    ax.set_xlabel("Annualized alpha (pp)", fontsize=11, color=TEXT)
    ax.set_ylabel("Density", fontsize=11, color=TEXT)
    add_legend(ax)
    fig.suptitle("Bootstrap",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    footnote(fig, f"{meta['n_iter']} resamples")

    path = "output/spx_consensus_stress_2_bootstrap.png"
    save_fig(fig, path)
    return fig, path
