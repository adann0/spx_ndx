"""Shared styling for consensus plots.

Re-exports colors and style helpers from spx_ndx.utils, plus
plot-specific helpers (_save, _annotate).
"""

import matplotlib.pyplot as plt

from spx_ndx.utils import (
    apply_style, add_legend, style_table,
    BG, GRID, BORDER, TEXT, DIM,
    BLUE, GREEN, RED, ORANGE, PURPLE, YELLOW, TEAL,
)

STAT = "#B0B8C1"

__all__ = [
    "apply_style", "add_legend", "style_table",
    "BG", "GRID", "BORDER", "TEXT", "DIM", "STAT",
    "BLUE", "GREEN", "RED", "ORANGE", "PURPLE", "YELLOW", "TEAL",
    "save_fig", "annotate_ax", "footnote", "stat_line",
]


def save_fig(fig, path):
    """Save figure and close it. Returns the path."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return path


def annotate_ax(ax, text, xy=(0.98, 0.02), va="bottom", ha="right", fontsize=10, **kw):
    """Add annotation box to axes."""
    ax.annotate(
        text, xy=xy, xycoords="axes fraction", fontsize=fontsize,
        va=va, ha=ha, color=TEXT,
        bbox=dict(boxstyle="round,pad=0.4", facecolor=BORDER, edgecolor=GRID, alpha=0.95),
        **kw,
    )


def footnote(fig, text):
    """Add italic footnote at bottom-left of figure."""
    fig.text(0.02, 0.01, text, fontsize=8, color=DIM, style="italic",
             va="bottom", ha="left")


def stat_line(ax, text):
    """Add stats annotation at bottom-left of axes."""
    ax.annotate(text, xy=(0.01, 0.03), xycoords="axes fraction",
                fontsize=8, color=STAT, va="bottom")
