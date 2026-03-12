"""Plot: Signal noise injection."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, footnote, TEXT, BLUE, RED, ORANGE


def plot_signal_noise(metrics, df, label):
    """CAGR degradation and distribution as signal noise is injected."""
    base = metrics["baseline"]
    signal_noise_metrics = metrics["signal_noise"]
    real_cagr = base["real_cagr"]
    bh_cagr = base["buy_hold_cagr"]
    noise_percents = signal_noise_metrics["noise_percents"]
    noise_res = signal_noise_metrics["results"]

    means = [np.mean(noise_res[str(p)]["cagrs"]) * 100 for p in noise_percents]
    stds = [np.std(noise_res[str(p)]["cagrs"]) * 100 for p in noise_percents]

    fig, ax = plt.subplots(figsize=(10, 6))
    apply_style(fig, ax)

    ax.fill_between(noise_percents,
                    [mu - 2*s for mu, s in zip(means, stds)],
                    [mu + 2*s for mu, s in zip(means, stds)],
                    color=BLUE, alpha=0.1, label="+-2sig")
    ax.fill_between(noise_percents,
                    [mu - s for mu, s in zip(means, stds)],
                    [mu + s for mu, s in zip(means, stds)],
                    color=BLUE, alpha=0.2, label="+-1sig")
    ax.plot(noise_percents, means, "o-", color=TEXT, lw=2, ms=6, label="Mean CAGR")
    ax.axhline(real_cagr * 100, color=RED, ls="-", lw=1.5, alpha=0.5,
               label="Real")
    ax.axhline(bh_cagr * 100, color=ORANGE, ls="--", lw=1.5,
               label="B&H")
    ax.set_xlabel("Signal noise (%)", fontsize=11, color=TEXT)
    ax.set_ylabel("CAGR %", fontsize=11, color=TEXT)
    add_legend(ax)

    fig.suptitle("Signal Noise Injection",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    footnote(fig, f"{signal_noise_metrics['n_trials']} trials per level")

    path = "output/spx_consensus_stress_3_noise.png"
    save_fig(fig, path)
    return fig, path
