"""Plot: Return noise injection."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, stat_line, TEXT, RED, ORANGE, PURPLE


def plot_return_noise(metrics, df, label):
    """CAGR stability under additive noise on SPX returns."""
    base = metrics["baseline"]
    return_noise_metrics = metrics["return_noise"]
    real_cagr = base["real_cagr"]
    bh_cagr = base["buy_hold_cagr"]
    noise_mults = return_noise_metrics["noise_mults"]
    return_noise_results = return_noise_metrics["results"]

    means = [np.mean(return_noise_results[str(mult)]["cagrs"]) * 100 for mult in noise_mults]
    stds = [np.std(return_noise_results[str(mult)]["cagrs"]) * 100 for mult in noise_mults]
    x_labels = [f"{mult*100:.0f}%" for mult in noise_mults]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)
    ax.fill_between(range(len(noise_mults)),
                    [mu - 2*s for mu, s in zip(means, stds)],
                    [mu + 2*s for mu, s in zip(means, stds)],
                    color=PURPLE, alpha=0.1, label="+-2sig range")
    ax.fill_between(range(len(noise_mults)),
                    [mu - s for mu, s in zip(means, stds)],
                    [mu + s for mu, s in zip(means, stds)],
                    color=PURPLE, alpha=0.2, label="+-1sig range")
    ax.plot(range(len(noise_mults)), means, "o-", color=PURPLE, lw=2, ms=7,
            label="Mean CAGR")
    ax.axhline(real_cagr * 100, color=RED, ls="-", lw=1.5, alpha=0.5,
               label="No noise")
    ax.axhline(bh_cagr * 100, color=ORANGE, ls="--", lw=2,
               label="B&H")
    ax.set_xticks(range(len(noise_mults)))
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(
        "Noise added to SPX returns (% of realized volatility)",
        fontsize=11, color=TEXT)
    ax.set_ylabel("CAGR %", fontsize=11, color=TEXT)
    fig.suptitle("Return Noise",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)

    _last_beat = max(
        (mult for mult in noise_mults
         if np.mean(return_noise_results[str(mult)]["cagrs"]) > bh_cagr),
        default=None,
    )
    if _last_beat is not None:
        _stats = f"mean_CAGR>B&H up to {_last_beat*100:.0f}% noise"
    else:
        _stats = "mean_CAGR<B&H at all noise levels"
    stat_line(ax, _stats)
    add_legend(ax)

    path = "output/spx_consensus_stress_4_retnoise.png"
    save_fig(fig, path)
    return fig, path
