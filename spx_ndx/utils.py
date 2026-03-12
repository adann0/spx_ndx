"""
utils.py
Shared style, loaders, and helpers for all spx-ndx plot scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
from matplotlib.patches import Patch
from matplotlib.table import Table

# PATHS
ROOT = Path(__file__).resolve().parents[1]
DATAS = ROOT / "datas"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

# STYLE - GitHub Dark
BG = "#0D1117"
GRID = "#21262D"
BORDER = "#30363D"
TEXT = "#E6EDF3"
DIM = "#8B949E"
BLUE = "#58A6FF"
GREEN = "#2DA44E"
RED = "#CF222E"
ORANGE = "#E3712A"
PURPLE = "#BC8CFF"
YELLOW = "#D29922"
TEAL = "#39D353"

# Crash periods: (label, start, end, type)
# type: "bubble" = valuation driven | "exo" = exogenous shock
CRASHES = [
    ("Black Monday",  "1987-08-01", "1987-12-01", "exo"),
    ("Dot-com",       "2000-03-01", "2002-10-01", "bubble"),
    ("GFC",           "2007-10-01", "2009-03-01", "exo"),
    ("COVID",         "2020-02-01", "2020-04-01", "exo"),
]
CRASH_COLORS = {"bubble": RED, "exo": "#3B82F6"}

# Long-history crash periods (for full-history plots since 1871)
CRASHES_H = [
    ("1929-32", "1929-09-03", "1932-07-08"),
    ("Dot-com", "2000-03-24", "2002-10-09"),
]

# NBER recession periods (modern era)
RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),
]

# STYLE HELPERS
def apply_style(fig: Figure, ax_or_axes: Axes | list[Axes] | None = None) -> None:
    """Apply dark GitHub style to figure and axes."""
    fig.patch.set_facecolor(BG)
    axes = ax_or_axes if ax_or_axes is not None else fig.get_axes()
    if not hasattr(axes, "__iter__"):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)
        ax.grid(True, color=GRID, alpha=0.4, linewidth=0.5)
        ax.spines["bottom"].set_edgecolor(BORDER)
        ax.spines["left"].set_edgecolor(BORDER)
        ax.spines["top"].set_edgecolor(BORDER)
        ax.spines["right"].set_edgecolor(BORDER)

def add_legend(ax: Axes, loc: str = "upper left", **kw: Any) -> Legend:
    leg = ax.legend(
        fontsize=9,
        facecolor=BORDER,
        edgecolor=BORDER,
        labelcolor=TEXT,
        loc=loc,
        **kw
    )
    return leg

def add_twinx(ax: Axes) -> Axes:
    """Create a styled secondary y-axis (twinx)."""
    ax2 = ax.twinx()
    ax2.set_facecolor(BG)
    ax2.tick_params(colors=DIM, labelsize=8)
    for sp in ax2.spines.values():
        sp.set_edgecolor(BORDER)
    return ax2

def merge_legends(ax1: Axes, ax2: Axes, loc: str = "upper left") -> None:
    """Merge legends from two twinx axes into ax1."""
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, fontsize=8, facecolor=BORDER,
               labelcolor=TEXT, loc=loc)

def add_stats(ax: Axes, s: pd.Series, label: str = "",
              pos: tuple[float, float] = (0.01, 0.03)) -> None:
    """Annotate min/max/mean/current in bottom-left corner."""
    v = s.dropna()
    cur = v.iloc[-1]
    pct = (v <= cur).mean() * 100
    as_of = v.index[-1].strftime("%Y-%m-%d")
    txt = (
        f"{label}  current={cur:.2f}  "
        f"mean={v.mean():.2f}  "
        f"min={v.min():.2f}  max={v.max():.2f}  "
        f"pct={pct:.0f}%  as_of={as_of}"
    )
    ax.annotate(txt, xy=pos, xycoords="axes fraction",
                fontsize=7.5, color=DIM, va="bottom")

def shade_crashes(ax: Axes, alpha: float = 0.15, label: bool = True) -> None:
    """Shade crash periods on ax."""
    for i, (name, start, end, ctype) in enumerate(CRASHES):
        color = CRASH_COLORS.get(ctype, RED)
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=color, zorder=1)
        if label:
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            y = 0.97 if i % 2 == 0 else 0.90
            ax.annotate(name, xy=(mid, y), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6.5,
                        color=color, fontweight="bold")

def add_crashes_h(ax: Axes) -> None:
    """Shade long-history crash periods (1929-32, Dot-com) on ax."""
    for label, start, end in CRASHES_H:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   color=RED, alpha=0.10, zorder=1, lw=0)
        ax.axvline(pd.Timestamp(start), color=RED,
                   lw=0.8, ls="--", alpha=0.35, zorder=2)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ylim = ax.get_ylim()
        y = ylim[0] + (ylim[1] - ylim[0]) * 0.96
        ax.text(mid, y, label, color=RED, fontsize=7.5,
                ha="center", va="top", alpha=0.80,
                style="italic", fontweight="bold")

def shade_recessions(ax: Axes, alpha: float = 0.15) -> None:
    """Shade NBER recession periods on ax."""
    for start, end in RECESSIONS:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=alpha, color=DIM, zorder=1)

def fmt_xaxis(ax: Axes, freq: str = "10YS") -> None:
    """Format x-axis with clean year labels."""
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center")

def save_fig(fig: Figure, filename: str, dpi: int = 150) -> Path:
    """Save figure to output/ directory."""
    path = OUTPUT / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  -> {filename}")
    return path

# LOADERS
def load_pq(name: str) -> pd.Series:
    """Load historical parquet (date+value or OHLCV close)."""
    df = pd.read_parquet(DATAS / f"{name}.parquet")
    if "date" in df.columns:
        return (df.sort_values("date")
                  .drop_duplicates("date")
                  .set_index("date")["value"]
                  .sort_index())
    return df["close"].sort_index()

def load_yahoo(name: str, col: str = "close") -> pd.Series:
    """Load Yahoo Finance parquet (DatetimeIndex, OHLCV)."""
    df = pd.read_parquet(DATAS / f"{name}.parquet")
    return df[col].rename(name)

def load_fred(name: str) -> pd.Series:
    """Load FRED parquet (date, value) -> Series with DatetimeIndex."""
    df = pd.read_parquet(DATAS / f"fred_{name}.parquet")
    return df.set_index("date")["value"].sort_index().rename(name)

def load_multpl(name: str) -> pd.Series:
    """Load multpl parquet (date, value) -> Series with DatetimeIndex."""
    df = pd.read_parquet(DATAS / f"{name}.parquet")
    return df.set_index("date")["value"].sort_index().rename(name)

def load_ohlcv(name: str) -> pd.DataFrame:
    """Load full OHLCV DataFrame from Yahoo parquet."""
    return pd.read_parquet(DATAS / f"{name}.parquet")

# COMMON TRANSFORMS
def resample_monthly(s: pd.Series) -> pd.Series:
    return s.resample("ME").last()

def rebase(s: pd.Series, base: float = 100) -> pd.Series:
    """Rebase series to base at first non-NaN value."""
    first = s.dropna().iloc[0]
    return s / first * base

def pct_rank_expanding(s: pd.Series, min_periods: int = 60) -> pd.Series:
    """Expanding window percentile rank (0-100)."""
    return s.expanding(min_periods=min_periods).rank(pct=True, method="max") * 100

def drawdown_from_ath(s: pd.Series) -> pd.Series:
    """Drawdown from all-time high (%)."""
    ath = s.expanding().max()
    return (s - ath) / ath * 100

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def rolling_cagr(s: pd.Series, years: int) -> pd.Series:
    """Annualized return over rolling N-year window."""
    n = years * 12  # assumes monthly
    return (s / s.shift(n)) ** (1 / years) - 1

# CHART HELPERS
def style_table(tbl: Table) -> None:
    """Apply dark GitHub style to a matplotlib table."""
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor(BORDER if row == 0 else BG)
        cell.set_text_props(color=TEXT)
        cell.set_edgecolor(GRID)

def plot_ema200(series: pd.Series, name: str, color_above: str,
                filename: str, title: str) -> None:
    """Render price + EMA 200 chart (log scale, colored by above/below)."""
    ema200 = ema(series, 200)
    above = series >= ema200
    dates = series.index

    fig, ax = plt.subplots(figsize=(16, 6))
    apply_style(fig, ax)

    for i in range(1, len(dates)):
        c = color_above if above.iloc[i] else DIM
        ax.semilogy([dates[i-1], dates[i]],
                    [series.iloc[i-1], series.iloc[i]],
                    color=c, lw=0.9, alpha=0.85)

    ax.semilogy(dates, ema200, color=ORANGE, lw=1.6, label="EMA 200", zorder=4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    ax.axhline(series.iloc[-1], color=color_above, lw=0.8, ls="--", alpha=0.7)

    patches = [
        Patch(color=color_above, label=f"{name} (above EMA 200)"),
        Patch(color=DIM,         label=f"{name} (below EMA 200)"),
        Patch(color=ORANGE,      label="EMA 200"),
    ]
    ax.legend(handles=patches, fontsize=9, facecolor=BORDER,
              edgecolor=BORDER, labelcolor=TEXT, loc="upper left")
    ax.set_ylabel("Price", color=TEXT, fontsize=10)
    fmt_xaxis(ax)
    add_stats(ax, series, label=name)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=TEXT, y=1.01)
    save_fig(fig, filename)

# VOLUME PROFILE
def plot_volume_profile(ohlcv: pd.DataFrame, ticker: str, filename: str,
                        days: int = 90, bins: int = 60) -> None:
    """Render candlestick chart + volume profile sidebar."""
    df = ohlcv.tail(days).copy()
    lo = df["low"].min()
    hi = df["high"].max()
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    bin_w = edges[1] - edges[0]

    vp = np.zeros(bins)
    for _, row in df.iterrows():
        span = row["high"] - row["low"]
        if span <= 0:
            continue
        overlap = np.minimum(edges[1:], row["high"]) - np.maximum(edges[:-1], row["low"])
        overlap = np.maximum(overlap, 0)
        vp     += (overlap / span) * row["volume"]

    poc_i = vp.argmax()
    poc = centers[poc_i]
    lo_i = hi_i = poc_i
    cum = vp[poc_i]
    target = vp.sum() * 0.70
    while cum < target and (lo_i > 0 or hi_i < bins - 1):
        add_lo = vp[lo_i - 1] if lo_i > 0 else 0
        add_hi = vp[hi_i + 1] if hi_i < bins - 1 else 0
        if add_lo >= add_hi and lo_i > 0:
            lo_i -= 1; cum += vp[lo_i]
        elif hi_i < bins - 1:
            hi_i += 1; cum += vp[hi_i]
        else:  # pragma: no cover  - unreachable with non-negative volumes
            break
    vah, val = centers[hi_i], centers[lo_i]

    fig = plt.figure(figsize=(16, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[4, 1], wspace=0.02)
    ax_c = fig.add_subplot(gs[0])
    ax_v = fig.add_subplot(gs[1], sharey=ax_c)
    apply_style(fig, [ax_c, ax_v])

    for _, row in df.iterrows():
        color = GREEN if row["close"] >= row["open"] else RED
        ax_c.plot([row.name, row.name], [row["low"], row["high"]],
                  color=color, lw=0.8, alpha=0.7)
        ax_c.plot([row.name, row.name], [row["open"], row["close"]],
                  color=color, lw=3.0, solid_capstyle="round")

    e20 = ema(df["close"], 20)
    ax_c.plot(df.index, e20, color=ORANGE, lw=1.3, label="EMA 20", alpha=0.9)

    for level, color, lbl in [
        (poc, YELLOW, "POC"),
        (vah, ORANGE, "VAH"),
        (val, BLUE,   "VAL"),
    ]:
        ax_c.axhline(level, color=color, lw=1.0, ls="--", alpha=0.85, label=lbl)

    ax_c.set_ylabel("Price", color=TEXT, fontsize=10)
    ax_c.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.setp(ax_c.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
    add_legend(ax_c, loc="upper left")
    add_stats(ax_c, df["close"], label="close")

    bar_colors = [
        YELLOW if i == poc_i else (BLUE if lo_i <= i <= hi_i else DIM)
        for i in range(bins)
    ]
    ax_v.barh(centers, vp, height=bin_w * 0.88, color=bar_colors, alpha=0.85)
    for level, color in [(poc, YELLOW), (vah, ORANGE), (val, BLUE)]:
        ax_v.axhline(level, color=color, lw=1.0, ls="--", alpha=0.8)
    ax_v.tick_params(labelleft=False)
    ax_v.set_xlabel("Volume", color=TEXT, fontsize=8)
    ax_v.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x/1e6:.0f}M"))

    fig.suptitle(
        f"{ticker} - Volume Profile ({days} days)",
        fontsize=12, fontweight="bold", color=TEXT, y=1.01
    )
    save_fig(fig, filename)
