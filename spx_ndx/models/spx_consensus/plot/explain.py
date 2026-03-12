"""Plot: Position explanation (SPX chart + agreement + signal importance + recap)."""

import os as _os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from ._style import (
    apply_style, save_fig, style_table,
    TEXT, DIM, GRID, BORDER, BG, BLUE, GREEN, RED,
)


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _load_explain_data(metrics, df, explain_path, dataset):
    """Load and prepare all data needed for the explain plot.

    Returns a dict, or None if data is missing.
    """
    _last12 = metrics.get("last_12", {})
    if not _last12 or not _last12.get("dates"):
        return None
    if not _os.path.exists(explain_path):
        return None

    meta = metrics["meta"]
    periods_per_year = meta["periods_per_year"]

    with open(explain_path) as _f:
        explain_data = json.load(_f)

    signal_names = explain_data["signal_names"]
    n_last_periods = min(int(periods_per_year), len(df))
    raw_last = df[[f"raw_{n}" for n in signal_names]].values[-n_last_periods:]
    agree_last = df["agreement"].values[-n_last_periods:]
    ex_dates = [d.strftime("%Y-%m") for d in df.index[-n_last_periods:]]
    n_pipelines = explain_data["n_pipelines"]
    structural_importance = np.array(explain_data.get("structural_importance", []))
    formula_val = explain_data.get("current_formula_value")
    last_12_signals = _last12["signals"]

    if dataset is None:
        _ds = pd.read_parquet("datas/dataset_monthly.parquet")
        _ds.index = pd.to_datetime(_ds.index)
    else:
        _ds = dataset
    spx = _ds["spx_close"]

    dt_idx = (pd.to_datetime([d + "-01" for d in ex_dates])
              + pd.offsets.MonthEnd(0))
    spx_12 = spx.reindex(dt_idx, method="nearest")

    current_position = "IN (SPX)" if last_12_signals[-1] == 1 else "OUT (T-Bill)"
    current_position_color = GREEN if last_12_signals[-1] == 1 else RED

    return {
        "signal_names": signal_names,
        "raw_last": raw_last,
        "agree_last": agree_last,
        "ex_dates": ex_dates,
        "n_pipelines": n_pipelines,
        "structural_importance": structural_importance,
        "formula_val": formula_val,
        "last_12_signals": last_12_signals,
        "dt_idx": dt_idx,
        "spx_12": spx_12,
        "current_position": current_position,
        "current_position_color": current_position_color,
    }


def _draw_spx_chart(ax, d):
    """Draw SPX line chart with IN/OUT zones."""
    dt_idx = d["dt_idx"]
    spx_12 = d["spx_12"]
    signals = d["last_12_signals"]

    ax.plot(dt_idx, spx_12.values, color=BLUE, linewidth=2, zorder=5)
    ax.scatter(dt_idx, spx_12.values, color=BLUE, s=30, zorder=6)

    for i in range(len(dt_idx)):
        if i == 0:
            x0 = dt_idx[0] - pd.Timedelta(days=15)
        else:
            x0 = dt_idx[i - 1] + (dt_idx[i] - dt_idx[i - 1]) / 2
        if i == len(dt_idx) - 1:
            x1 = dt_idx[-1] + pd.Timedelta(days=15)
        else:
            x1 = dt_idx[i] + (dt_idx[i + 1] - dt_idx[i]) / 2
        c = GREEN if signals[i] == 1 else RED
        ax.axvspan(x0, x1, alpha=0.08, color=c, zorder=0)

    ax.set_ylabel("S&P 500", fontsize=11, color=TEXT)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.tick_params(axis="x", rotation=30)

    for i in range(len(dt_idx)):
        lbl = "IN" if signals[i] == 1 else "OUT"
        c = GREEN if signals[i] == 1 else RED
        ax.annotate(lbl, (dt_idx[i], spx_12.values[i]),
                    textcoords="offset points", xytext=(0, 12),
                    fontsize=8, fontweight="bold", color=c, ha="center")


def _draw_barh(ax, d, top_n=15):
    """Draw horizontal bar chart of top signals."""
    structural_importance = d["structural_importance"]
    signal_names = d["signal_names"]
    raw_last = d["raw_last"]

    if len(structural_importance) == 0:
        return

    order = np.argsort(structural_importance)[::-1]
    top_n = min(top_n, len(structural_importance))
    top_idx = order[:top_n][::-1]

    names_top = [signal_names[j] for j in top_idx]
    importance_top = structural_importance[top_idx]
    states = raw_last[-1]
    bar_colors = [GREEN if states[j] == 1 else RED for j in top_idx]

    ax.barh(range(top_n), importance_top, color=bar_colors, height=0.7,
            edgecolor=GRID, linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_top, fontsize=9)
    ax.set_xlabel("Structural importance (%)", fontsize=10, color=TEXT)

    ax.legend(
        handles=[Patch(facecolor=GREEN, label="ON"),
                 Patch(facecolor=RED, label="OFF")],
        loc="lower right", fontsize=9, framealpha=0.8,
        facecolor=BORDER, edgecolor=GRID, labelcolor=TEXT)


# ─── Main plot ────────────────────────────────────────────────────────────────

def plot_explain(metrics, df, label, explain_path="output/spx_consensus_explainability.json",
                 dataset=None):
    """2x2: SPX | pipeline agreement | barh signals | recap table."""
    d = _load_explain_data(metrics, df, explain_path, dataset)
    if d is None:
        return None, None

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    apply_style(fig, axes.flatten())

    # Top-left: SPX
    _draw_spx_chart(axes[0, 0], d)

    # Top-right: pipeline agreement over last 12 months
    ax_agree = axes[0, 1]
    n_pipelines = d["n_pipelines"]
    agree_last = d["agree_last"]
    ex_dates = d["ex_dates"]
    x = range(len(ex_dates))
    colors_agree = [GREEN if d["last_12_signals"][i] == 1 else RED
                    for i in range(len(ex_dates))]
    ax_agree.bar(x, agree_last, color=colors_agree, alpha=0.8, edgecolor=GRID,
                 linewidth=0.5)
    ax_agree.axhline(n_pipelines / 2, color=DIM, ls="--", lw=1, alpha=0.5)
    ax_agree.set_xticks(list(x))
    ax_agree.set_xticklabels(ex_dates, rotation=45, ha="right", fontsize=8)
    ax_agree.set_ylabel(f"Agreement (/{n_pipelines})", fontsize=10, color=TEXT)
    ax_agree.set_ylim(0, n_pipelines + 0.5)
    for i, v in enumerate(agree_last):
        ax_agree.text(i, v + 0.15, str(int(v)), ha="center", fontsize=8,
                      color=colors_agree[i], fontweight="bold")

    # Bottom-left: barh
    _draw_barh(axes[1, 0], d)

    # Bottom-right: recap table
    ax_recap = axes[1, 1]
    ax_recap.axis("off")
    raw_last = d["raw_last"]
    signal_names = d["signal_names"]
    signals_on = int(raw_last[-1].sum())
    n_total = len(signal_names)
    agreement = int(agree_last[-1])
    formula_val = d["formula_val"]

    recap_data = [
        ["Position", d["current_position"]],
        ["Signals ON", f"{signals_on}/{n_total}"],
        ["Pipeline agreement", f"{agreement}/{n_pipelines}"],
        ["Consensus", f"{formula_val:.3f}" if formula_val is not None else "N/A"],
    ]
    tbl = ax_recap.table(cellText=recap_data, cellLoc="center", loc="center",
                         colWidths=[0.5, 0.4])
    style_table(tbl)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2.5)
    tbl[0, 1].set_text_props(color=d["current_position_color"], fontweight="bold",
                              fontsize=14)

    fig.suptitle("Signals",
                 fontsize=15, fontweight="bold",
                 color=d["current_position_color"], y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    path = "output/spx_consensus_explain.png"
    save_fig(fig, path)
    return fig, path
