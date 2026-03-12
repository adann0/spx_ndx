"""Plot: Signal proximity - how close each indicator is to flipping.

Six visualizations of the same data:
  0. Horizontal bar chart (original)
  1. Gauge / speedometer small multiples
  2. Heatmap matrix (last 12 months × signals)
  3. Radar / spider chart
  4. Bubble scatter (distance × importance)
  5. Thermometer / bullet bars (historical context)
"""

import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.colors import LinearSegmentedColormap

from ._style import apply_style, save_fig, TEXT, DIM, GREEN, RED, BG, GRID, BORDER


# ─── Reverse mapping: template -> (column, direction, transform) ─────────────

_THRESHOLD_META = {
    "VIX<X":        ("vix_close", "below", None),
    "RSI<X":        ("rsi_14", "below", None),
    "RSI>X":        ("rsi_14", "above", None),
    "MACDhist>X":   ("macd_hist", "above", None),
    "MACDline>X":   ("macd_line", "above", None),
    "CAPE z<X":     ("cape_zscore", "below", None),
    "PE z<X":       ("pe_zscore", "below", None),
    "Compo<X":      ("composite_valuation", "below", None),
    "CAPE-PE<X":    ("cape_pe_spread", "below", None),
    "CPI<X":        ("cpi_yoy", "below", None),
    "EMA200>X":     ("spx_ema200_ratio", "above", lambda v: v / 100),
    "Gold/SPX z<X": ("gold_spx_ratio_zscore", "below", None),
    "YC z>X":       ("yield_curve_zscore", "above", None),
    "Credit z<X":   ("credit_spread_zscore", "below", None),
    "CuAu z>X":     ("copper_gold_zscore", "above", None),
    "RVol z<X":     ("realized_vol_zscore", "below", None),
    "BB>X":         ("bb_position", "above", None),
    "Ichi>X":       ("ichimoku_cloud_position", "above", None),
    "RSI z<X":      ("rsi_zscore", "below", None),
    "BBw<X":        ("bb_width", "below", None),
    "SAR>X":        ("sar_bullish", "above", None),
    "KST>X":        ("kst_diff", "above", None),
    "VolSpread<X":  ("vol_spread", "below", None),
    "Buffett z<X":  ("buffett_zscore", "below", None),
    "ECY z>X":      ("ecy_zscore", "above", None),
    "EY-10Y>X":     ("ey_10y_spread", "above", None),
    "Unemp<X":      ("unemployment", "below", None),
    "NFCI<X":       ("nfci", "below", None),
    "ANFCI z<X":    ("anfci_zscore", "below", None),
    "USD z<X":      ("dollar_major_zscore", "below", None),
    "CuAu mom>X":   ("copper_gold_mom_12m", "above", None),
    "Oil mom>X":    ("oil_mom_12m", "above", None),
    "RUT/SPX mom>X": ("rut_spx_mom", "above", None),
    "NDX/SPX mom>X": ("ndx_spx_mom", "above", None),
    "VIX z<X":      ("vix_zscore", "below", None),
    "DD>X":         ("spx_drawdown_from_ath", "above", None),
}

_BAND_META = {
    "RSI 30-X": ("rsi_14", 30),
}

_DIFF_META = {
    "SMA Xm":    ("sma_{}m", "spx_close"),
    "VP>VAL Xm": ("vp_val_{}m", "spx_close"),
}

_ALL_TEMPLATES = list(_THRESHOLD_META) + list(_BAND_META) + list(_DIFF_META)

_PROX_CMAP = LinearSegmentedColormap.from_list("prox", [RED, "#1a1a2e", GREEN], N=256)


# ─── Shared helpers ──────────────────────────────────────────────────────────

def _parse_signal_name(name):
    """Parse signal name -> (template, value)."""
    for template in _ALL_TEMPLATES:
        idx = template.rfind("X")
        prefix = template[:idx]
        suffix = template[idx + 1:]
        if name.startswith(prefix) and name.endswith(suffix):
            val_str = name[len(prefix):len(name) - len(suffix)] if suffix else name[len(prefix):]
            try:
                return template, float(val_str) if "." in val_str else int(val_str)
            except ValueError:
                continue
    return None, None


def _get_signal_meta(name):
    """Get metadata dict for a signal name, or None."""
    template, value = _parse_signal_name(name)
    if template is None:
        return None
    if template in _THRESHOLD_META:
        col, direction, transform = _THRESHOLD_META[template]
        return {"type": "threshold", "col": col,
                "thresh": transform(value) if transform else value,
                "direction": direction}
    if template in _BAND_META:
        col, lower = _BAND_META[template]
        return {"type": "band", "col": col, "lower": lower, "upper": float(value)}
    if template in _DIFF_META:
        col_pattern, ref_col = _DIFF_META[template]
        return {"type": "diff", "col": col_pattern.format(value), "ref_col": ref_col}
    return None


def _distance_at(meta, dataset, idx):
    """Compute (distance, is_on) at dataset.iloc[idx]. distance > 0 -> ON."""
    t = meta["type"]
    if t == "threshold":
        col = meta["col"]
        if col not in dataset.columns:
            return np.nan, False
        val = float(dataset[col].iloc[idx])
        if np.isnan(val):
            return np.nan, False
        if meta["direction"] == "below":
            return meta["thresh"] - val, val < meta["thresh"]
        return val - meta["thresh"], val > meta["thresh"]
    if t == "band":
        col = meta["col"]
        if col not in dataset.columns:
            return np.nan, False
        val = float(dataset[col].iloc[idx])
        if np.isnan(val):
            return np.nan, False
        lo, hi = meta["lower"], meta["upper"]
        if val <= lo:
            return val - lo, False
        if val >= hi:
            return hi - val, False
        return min(val - lo, hi - val), True
    if t == "diff":
        col, ref = meta["col"], meta["ref_col"]
        if col not in dataset.columns or ref not in dataset.columns:
            return np.nan, False
        v, r = float(dataset[col].iloc[idx]), float(dataset[ref].iloc[idx])
        if np.isnan(v) or np.isnan(r):
            return np.nan, False
        d = r - v
        return d, d > 0
    return np.nan, False


def _fmt_val(val):
    """Format a value for display."""
    if isinstance(val, str):
        return val
    if abs(val) >= 100:
        return f"{val:,.0f}"
    if abs(val) >= 10:
        return f"{val:.1f}"
    return f"{val:.2f}"


def _threshold_display(meta):
    if meta["type"] == "threshold":
        return meta["thresh"]
    if meta["type"] == "band":
        return f"{meta['lower']}-{meta['upper']}"
    return 0


def _current_value(meta, dataset, idx=-1):
    """Raw indicator value at index."""
    if meta["type"] in ("threshold", "band"):
        col = meta["col"]
        return float(dataset[col].iloc[idx]) if col in dataset.columns else np.nan
    col, ref = meta["col"], meta["ref_col"]
    if col not in dataset.columns or ref not in dataset.columns:
        return np.nan
    return float(dataset[ref].iloc[idx]) - float(dataset[col].iloc[idx])


def _load_proximity_data(explain_path, dataset):
    """Shared data loader for all proximity plots.

    Returns (rows, position_label, position_color) or (None, None, None).
    rows sorted by |distance| ascending (closest to flip first).
    Each row: name, meta, current, threshold, distance, is_on, importance, norm_distance.
    """
    if not os.path.exists(explain_path):
        return None, None, None
    with open(explain_path) as f:
        explain = json.load(f)

    signal_names = explain["signal_names"]
    importance_arr = np.array(explain.get("structural_importance", []))

    rows = []
    for i, name in enumerate(signal_names):
        meta = _get_signal_meta(name)
        if meta is None:
            continue
        dist, is_on = _distance_at(meta, dataset, -1)
        if np.isnan(dist):
            continue
        # Normalize using historical range (last 60 months)
        n_hist = min(60, len(dataset))
        hist = [d for j in range(-n_hist, 0)
                for d in [_distance_at(meta, dataset, j)[0]] if not np.isnan(d)]
        if hist:
            scale = max(abs(min(hist)), abs(max(hist)), 1e-9)
            norm = np.clip(dist / scale, -1, 1)
        else:
            norm = 0.0

        rows.append({
            "name": name, "meta": meta,
            "current": _current_value(meta, dataset),
            "threshold": _threshold_display(meta),
            "distance": dist, "is_on": is_on,
            "importance": float(importance_arr[i]) if i < len(importance_arr) else 0,
            "norm_distance": norm,
        })

    if not rows:
        return None, None, None
    rows.sort(key=lambda r: abs(r["distance"]))
    return rows, None, None


def _title_parts(df, rows):
    """Return (position_str, color, n_on, n_total)."""
    last = df["signal"].values[-1] if "signal" in df.columns else None
    pos = "IN (SPX)" if last == 1 else "OUT (T-Bill)" if last is not None else "?"
    col = GREEN if last == 1 else RED
    return pos, col, sum(r["is_on"] for r in rows), len(rows)


# ─── Plot 0: Horizontal bar chart ───────────────────────────────────────────

def plot_proximity(metrics, df, label,
                   explain_path="output/spx_consensus_explainability.json",
                   dataset=None):
    """Horizontal diverging bars: distance to threshold per signal."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    n = len(rows)
    fig, ax = plt.subplots(figsize=(12, max(4, 0.5 * n + 1.5)))
    apply_style(fig, ax)

    y_pos = np.arange(n)
    colors = [GREEN if r["is_on"] else RED for r in rows]
    ax.barh(y_pos, [r["distance"] for r in rows], color=colors,
            height=0.65, edgecolor=GRID, linewidth=0.5, alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['name']}  ({r['importance']:.0f}%)" for r in rows], fontsize=9)
    ax.axvline(0, color=DIM, linewidth=1.5, alpha=0.6)
    ax.invert_yaxis()

    for i, r in enumerate(rows):
        txt = f" {_fmt_val(r['current'])} -> {_fmt_val(r['threshold'])}  ({r['distance']:+.2f})"
        ha = "left" if r["distance"] >= 0 else "right"
        ax.text(r["distance"], i, txt, va="center", ha=ha, fontsize=8,
                color=TEXT, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, alpha=0.8, edgecolor="none"))

    ax.set_xlabel("Distance to threshold (<- flip zone | margin ->)", fontsize=10, color=TEXT)
    pos, pc, n_on, n_tot = _title_parts(df, rows)
    ax.set_title(f"Signal Proximity  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=13, fontweight="bold", color=pc, pad=15)
    plt.tight_layout()
    path = "output/spx_consensus_proximity.png"
    save_fig(fig, path)
    return fig, path


# ─── Plot 1: Gauge / speedometer ────────────────────────────────────────────

def _draw_gauge(ax, row):
    """Draw a single semicircle gauge on a regular Axes."""
    ax.set_xlim(-1.4, 1.4)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colored semicircle ring: red left, green right
    for angle_start, angle_end, c in [(90, 180, RED), (0, 90, GREEN)]:
        ax.add_patch(Wedge((0, 0), 1.0, angle_start, angle_end,
                           width=0.3, facecolor=c, alpha=0.2, edgecolor="none"))

    # Threshold tick at top (90°)
    ax.plot([0, 0], [0.72, 1.02], color=DIM, lw=1, ls="--", alpha=0.5)

    # Needle: norm_distance in [-1, 1] -> angle in [π, 0]
    norm = np.clip(row["norm_distance"], -1, 1)
    angle = np.pi * (1 - norm) / 2
    nx, ny = 0.88 * np.cos(angle), 0.88 * np.sin(angle)
    color = GREEN if row["is_on"] else RED
    ax.plot([0, nx], [0, ny], color=color, lw=2.5, solid_capstyle="round", zorder=5)
    ax.scatter([0], [0], color=TEXT, s=15, zorder=6)

    # Labels
    ax.text(0, -0.15, row["name"], ha="center", va="top",
            fontsize=8, color=color, fontweight="bold")
    ax.text(0, -0.35, f"{row['importance']:.0f}%  |  {_fmt_val(row['current'])}",
            ha="center", va="top", fontsize=7, color=DIM)


def plot_proximity_gauges(metrics, df, label,
                          explain_path="output/spx_consensus_explainability.json",
                          dataset=None):
    """Small multiples of semicircle gauges."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    # Sort by importance descending for gauges
    rows_imp = sorted(rows, key=lambda r: -r["importance"])
    n = len(rows_imp)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    apply_style(fig, axes.flatten())

    for i, r in enumerate(rows_imp):
        _draw_gauge(axes[i // ncols, i % ncols], r)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    pos, pc, n_on, n_tot = _title_parts(df, rows_imp)
    fig.suptitle(f"Signal Gauges  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=14, fontweight="bold", color=pc, y=1.01)
    plt.tight_layout()
    path = "output/spx_consensus_proximity_gauges.png"
    save_fig(fig, path)
    return fig, path


# ─── Plot 2: Heatmap (last 12 months × signals) ─────────────────────────────

def plot_proximity_heatmap(metrics, df, label,
                           explain_path="output/spx_consensus_explainability.json",
                           dataset=None):
    """Heatmap: normalized distance per signal over the last 12 months."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    # Sort by importance descending
    rows_imp = sorted(rows, key=lambda r: -r["importance"])
    n_months = 12
    n_sig = len(rows_imp)

    matrix = np.full((n_sig, n_months), np.nan)
    n_ds = len(dataset)
    start = max(0, n_ds - n_months)
    date_labels = []
    for t_idx, ds_idx in enumerate(range(start, n_ds)):
        dt = dataset.index[ds_idx]
        date_labels.append(dt.strftime("%Y-%m"))
        for s_idx, r in enumerate(rows_imp):
            d, _ = _distance_at(r["meta"], dataset, ds_idx)
            matrix[s_idx, t_idx] = d

    # Normalize each row by its historical scale (same as norm_distance logic)
    norm_matrix = np.zeros_like(matrix)
    for s_idx, r in enumerate(rows_imp):
        row_vals = matrix[s_idx]
        valid = row_vals[~np.isnan(row_vals)]
        if len(valid) == 0:
            continue
        scale = max(abs(valid.min()), abs(valid.max()), 1e-9)
        norm_matrix[s_idx] = np.clip(matrix[s_idx] / scale, -1, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_months * 0.9), max(5, n_sig * 0.45)))
    apply_style(fig, ax)

    im = ax.imshow(norm_matrix, aspect="auto", cmap=_PROX_CMAP, vmin=-1, vmax=1,
                   interpolation="nearest")

    ax.set_xticks(range(len(date_labels)))
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n_sig))
    ax.set_yticklabels([f"{r['name']} ({r['importance']:.0f}%)" for r in rows_imp], fontsize=9)

    # Annotate cells with ON/OFF
    for si in range(n_sig):
        for ti in range(len(date_labels)):
            val = norm_matrix[si, ti]
            if np.isnan(val):
                continue
            txt = "ON" if val > 0 else "OFF"
            c = "#ffffff" if abs(val) > 0.4 else DIM
            ax.text(ti, si, txt, ha="center", va="center", fontsize=6, color=c)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("<- OFF | ON ->", fontsize=9, color=TEXT)
    cbar.ax.tick_params(labelcolor=TEXT, labelsize=7)

    pos, pc, n_on, n_tot = _title_parts(df, rows_imp)
    ax.set_title(f"Signal Proximity Heatmap  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=13, fontweight="bold", color=pc, pad=15)
    plt.tight_layout()
    path = "output/spx_consensus_proximity_heatmap.png"
    save_fig(fig, path)
    return fig, path


# ─── Plot 3: Radar / spider chart ───────────────────────────────────────────

def plot_proximity_radar(metrics, df, label,
                         explain_path="output/spx_consensus_explainability.json",
                         dataset=None):
    """Polar radar chart: each axis is a signal, radius = normalized proximity."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    # Sort by importance descending
    rows_imp = sorted(rows, key=lambda r: -r["importance"])
    n = len(rows_imp)
    if n < 3:
        return None, None

    # Map norm_distance [-1, 1] -> radius [0, 1] (0 = far OFF, 0.5 = threshold, 1 = far ON)
    radii = [(r["norm_distance"] + 1) / 2 for r in rows_imp]
    names = [r["name"] for r in rows_imp]

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    radii_closed = radii + [radii[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)

    # Fill
    ax.fill(angles_closed, radii_closed, alpha=0.15, color=GREEN)
    ax.plot(angles_closed, radii_closed, color=GREEN, lw=2)

    # Threshold circle at 0.5
    theta_circle = np.linspace(0, 2 * np.pi, 100)
    ax.plot(theta_circle, [0.5] * 100, color=DIM, ls="--", lw=1, alpha=0.5)

    # Points
    for i, r in enumerate(rows_imp):
        c = GREEN if r["is_on"] else RED
        ax.scatter([angles[i]], [radii[i]], color=c, s=50, zorder=5, edgecolors=TEXT, linewidths=0.5)

    ax.set_xticks(angles)
    ax.set_xticklabels([f"{nm}\n({rows_imp[i]['importance']:.0f}%)"
                        for i, nm in enumerate(names)],
                       fontsize=7, color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["OFF", "", "threshold", "", "ON"], fontsize=7, color=DIM)
    ax.tick_params(colors=DIM)
    ax.spines["polar"].set_color(GRID)
    ax.grid(color=GRID, alpha=0.3)

    pos, pc, n_on, n_tot = _title_parts(df, rows_imp)
    ax.set_title(f"Signal Radar  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=13, fontweight="bold", color=pc, pad=25)
    plt.tight_layout()
    path = "output/spx_consensus_proximity_radar.png"
    save_fig(fig, path)
    return fig, path


# ─── Plot 4: Bubble scatter (distance × importance) ─────────────────────────

def plot_proximity_bubble(metrics, df, label,
                          explain_path="output/spx_consensus_explainability.json",
                          dataset=None):
    """Scatter: X = normalized distance, Y = structural importance, size = importance."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    fig, ax = plt.subplots(figsize=(12, 7))
    apply_style(fig, ax)

    xs = [r["norm_distance"] for r in rows]
    ys = [r["importance"] for r in rows]
    sizes = [r["importance"] * 30 + 40 for r in rows]
    colors = [GREEN if r["is_on"] else RED for r in rows]

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors=TEXT, linewidths=0.5, zorder=5)

    # Threshold line at x=0
    ax.axvline(0, color=DIM, ls="--", lw=1.5, alpha=0.5)

    # Quadrant labels
    ax.text(-0.5, max(ys) * 0.95, "OFF + Important\n(risk zone)", fontsize=9,
            color=RED, ha="center", va="top", alpha=0.6, fontweight="bold")
    ax.text(0.5, max(ys) * 0.95, "ON + Important\n(safe zone)", fontsize=9,
            color=GREEN, ha="center", va="top", alpha=0.6, fontweight="bold")

    # Label each point
    for r, x, y in zip(rows, xs, ys):
        ax.annotate(r["name"], (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=7, color=TEXT)

    ax.set_xlabel("Normalized distance (<- OFF | threshold | ON ->)", fontsize=10, color=TEXT)
    ax.set_ylabel("Structural importance (%)", fontsize=10, color=TEXT)
    ax.set_xlim(-1.15, 1.15)

    pos, pc, n_on, n_tot = _title_parts(df, rows)
    ax.set_title(f"Signal Risk Map  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=13, fontweight="bold", color=pc, pad=15)
    plt.tight_layout()
    path = "output/spx_consensus_proximity_bubble.png"
    save_fig(fig, path)
    return fig, path


# ─── Plot 5: Thermometer / bullet bars ──────────────────────────────────────

def plot_proximity_thermo(metrics, df, label,
                          explain_path="output/spx_consensus_explainability.json",
                          dataset=None):
    """Bullet chart: historical range + threshold + current value per signal."""
    if dataset is None:
        dataset = pd.read_parquet("datas/dataset_monthly.parquet")
        dataset.index = pd.to_datetime(dataset.index)

    rows, _, _ = _load_proximity_data(explain_path, dataset)
    if rows is None:
        return None, None

    # Sort by importance descending
    rows_imp = sorted(rows, key=lambda r: -r["importance"])
    n = len(rows_imp)

    n_hist = min(60, len(dataset))
    for r in rows_imp:
        vals = []
        for j in range(-n_hist, 0):
            v = _current_value(r["meta"], dataset, j)
            if not np.isnan(v):
                vals.append(v)
        if vals:
            r["hist_min"] = min(vals)
            r["hist_max"] = max(vals)
            r["hist_p25"] = float(np.percentile(vals, 25))
            r["hist_p75"] = float(np.percentile(vals, 75))
        else:
            r["hist_min"] = r["hist_max"] = r["current"]
            r["hist_p25"] = r["hist_p75"] = r["current"]

    fig, ax = plt.subplots(figsize=(14, max(4, 0.55 * n + 1)))
    apply_style(fig, ax)

    y_pos = np.arange(n)
    bar_height = 0.6

    for i, r in enumerate(rows_imp):
        lo, hi = r["hist_min"], r["hist_max"]
        span = hi - lo if hi != lo else 1
        p25, p75 = r["hist_p25"], r["hist_p75"]

        # Full range bar (light gray)
        ax.barh(i, span, left=lo, height=bar_height, color=GRID, alpha=0.25, edgecolor="none")
        # IQR band (darker)
        ax.barh(i, p75 - p25, left=p25, height=bar_height, color=GRID, alpha=0.35, edgecolor="none")

        # Threshold marker
        thresh = r["threshold"]
        if isinstance(thresh, (int, float)):
            ax.plot([thresh, thresh], [i - bar_height / 2, i + bar_height / 2],
                    color=DIM, lw=2, zorder=4)

        # Current value dot
        color = GREEN if r["is_on"] else RED
        ax.scatter([r["current"]], [i], color=color, s=80, zorder=5,
                   edgecolors=TEXT, linewidths=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['name']}  ({r['importance']:.0f}%)" for r in rows_imp], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Indicator value (gray = 5y range, bar = IQR, dot = current, line = threshold)",
                  fontsize=8, color=DIM)

    pos, pc, n_on, n_tot = _title_parts(df, rows_imp)
    ax.set_title(f"Signal Thermometer  -  {pos}  ({n_on}/{n_tot} ON)",
                 fontsize=13, fontweight="bold", color=pc, pad=15)
    plt.tight_layout()
    path = "output/spx_consensus_proximity_thermo.png"
    save_fig(fig, path)
    return fig, path
