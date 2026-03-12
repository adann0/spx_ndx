"""Plot: Rolling Sharpe."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, stat_line, TEXT, DIM, GREEN, RED


def plot_rolling_rtr(metrics, df, label):
    """Rolling 36-month Sharpe for strategy vs B&H."""
    if "rolling_rtr" not in metrics or metrics["rolling_rtr"] is None:
        return None, None

    base = metrics["baseline"]
    real_sharpe = base["real_sharpe"]
    rolling_data = metrics["rolling_rtr"]
    rolling_dates = pd.to_datetime(rolling_data["dates"])

    sharpe_strategy = np.array(rolling_data["strategy_sharpe"])
    sharpe_buy_hold = np.array(rolling_data["buy_hold_sharpe"])
    sharpe_percent_above_zero = rolling_data.get(
        "percent_sharpe_above_zero",
        float((sharpe_strategy > 0).mean() * 100))
    sharpe_percent_above_buy_hold = rolling_data.get(
        "percent_sharpe_above_buy_hold",
        float((sharpe_strategy > sharpe_buy_hold).mean() * 100))
    sharpe_min = rolling_data.get("sharpe_min", float(np.min(sharpe_strategy)))
    sharpe_max = rolling_data.get("sharpe_max", float(np.max(sharpe_strategy)))

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, ax)

    ax.fill_between(rolling_dates, sharpe_strategy, 0,
                    where=sharpe_strategy >= 0, color=GREEN, alpha=0.15,
                    interpolate=True)
    ax.fill_between(rolling_dates, sharpe_strategy, 0,
                    where=sharpe_strategy < 0, color=RED, alpha=0.15,
                    interpolate=True)
    ax.plot(rolling_dates, sharpe_strategy, color=GREEN, lw=1.8, label="Strategy",
            zorder=3)
    ax.plot(rolling_dates, sharpe_buy_hold, color=DIM, lw=1.2, ls="--", label="B&H",
            zorder=2)
    ax.axhline(0, color=TEXT, lw=0.5, ls="-", alpha=0.3)
    ax.axhline(real_sharpe, color=GREEN, lw=1, ls=":", alpha=0.5,
               label="Full-period Sharpe")

    stat_line(ax, f"sharpe>0={sharpe_percent_above_zero:.0f}%  "
                  f"sharpe>B&H={sharpe_percent_above_buy_hold:.0f}%  "
                  f"min={sharpe_min:.2f}  max={sharpe_max:.2f}  "
                  f"full_period={real_sharpe:.2f}")

    ax.set_ylabel("Rolling 3Y Sharpe", color=TEXT, fontsize=11)
    add_legend(ax)
    fig.suptitle("Rolling Sharpe (36 Months)",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    plt.tight_layout()
    path = "output/spx_consensus_stress_10_rolling_sharpe.png"
    save_fig(fig, path)

    return fig, path
