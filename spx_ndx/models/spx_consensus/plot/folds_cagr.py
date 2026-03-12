"""Plot: Walk-forward folds CAGR bar chart."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, add_legend, save_fig, TEXT, DIM, GREEN, ORANGE


def plot_folds_cagr(metrics, df, label,
                   explain_path="output/spx_consensus_explainability.json"):
    """Strategy vs B&H CAGR bar chart per walk-forward fold."""
    import json, os as _os
    if not _os.path.exists(explain_path):
        return None, None

    with open(explain_path) as f:
        explain_data = json.load(f)
    _folds = explain_data.get("fold_results", [])
    if not _folds:
        return None, None

    _periods = [r["period"] for r in _folds]
    _strategy_cagrs = [r["test_cagr"] * 100 for r in _folds]
    _buy_hold_cagrs = [r["buy_hold_cagr"] * 100 for r in _folds]

    fig, ax = plt.subplots(figsize=(14, 6))
    apply_style(fig, [ax])

    x = np.arange(len(_periods))
    bar_width = 0.35
    ax.bar(x - bar_width/2, _strategy_cagrs, bar_width, color=GREEN, label="Strategy", alpha=0.9)
    ax.bar(x + bar_width/2, _buy_hold_cagrs, bar_width, color=ORANGE, label="B&H", alpha=0.7)
    ax.axhline(0, color=DIM, linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(_periods, rotation=45, ha="right")
    ax.set_ylabel("CAGR (%)", color=TEXT)
    ax.set_title("Walk-Forward CAGR",
                 fontsize=13, fontweight="bold", color=TEXT)
    add_legend(ax, loc="upper left")

    plt.tight_layout()

    path = "output/spx_consensus_folds_cagr.png"
    save_fig(fig, path)
    return fig, path
