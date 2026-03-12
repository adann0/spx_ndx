"""cape_scatter_10y.py - CAPE Implied 10Y Return (Scatter) -> output/cape_scatter_10y.png"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx_h = u.load_pq("spx_historical_prices")
cape_h = u.load_pq("spx_shiller_pe_ratio")

HORIZON_SC = 120
START_SC = pd.Timestamp("1990-01-01")

cape_a_sc, spx_a_sc = cape_h.align(spx_h, join="inner")
fwd_raw = spx_a_sc.shift(-HORIZON_SC) / spx_a_sc
fwd = (fwd_raw ** (1/10) - 1) * 100

common_sc = cape_a_sc.index.intersection(fwd.dropna().index)
cape_sel = cape_a_sc.reindex(common_sc)
fwd_sel = fwd.reindex(common_sc)
mask_sc = (np.isfinite(cape_sel.values) & np.isfinite(fwd_sel.values)
             & (common_sc >= START_SC))

x_sc = cape_sel.values[mask_sc]
y_sc = fwd_sel.values[mask_sc]
d_sc = common_sc[mask_sc]

slope_sc, intercept_sc, r_sc, pval_sc, se_sc = scipy_stats.linregress(x_sc, y_sc)
r2_sc = r_sc ** 2
x_line_sc = np.linspace(x_sc.min(), x_sc.max() * 1.05, 300)
y_line_sc = slope_sc * x_line_sc + intercept_sc

cur_cape_sc = cape_h.iloc[-1]
cur_implied_sc = slope_sc * cur_cape_sc + intercept_sc

print(f"  Post-1990  n={len(x_sc)}  R²={r2_sc:.3f}  p={pval_sc:.2e}")
print(f"  Current CAPE: {cur_cape_sc:.1f}  ->  Implied 10Y CAGR: {cur_implied_sc:.1f}%/yr")


def decade_color_sc(dt):
    y = dt.year
    if y < 2000: return u.BLUE
    if y < 2010: return u.RED
    if y < 2020: return u.GREEN
    return u.YELLOW


pt_colors_sc = [decade_color_sc(dt) for dt in d_sc]

fig, ax = plt.subplots(figsize=(14, 9))
u.apply_style(fig, ax)

ax.scatter(x_sc, y_sc, c=pt_colors_sc, alpha=0.55, s=22, zorder=3)
ax.plot(x_line_sc, y_line_sc, color=u.ORANGE, lw=2.2, zorder=4,
        label=f"OLS  slope={slope_sc:.2f}  R²={r2_sc:.3f}  p={pval_sc:.1e}  n={len(x_sc)}")
ax.axhline(0, color=u.DIM, lw=0.8, ls="--", alpha=0.5, label="Return = 0%")

n_sc = len(x_sc)
x_mean_sc = x_sc.mean()
s_err_sc = se_sc * np.sqrt(1 + 1/n_sc + (x_line_sc - x_mean_sc)**2
                             / ((x_sc - x_mean_sc)**2).sum())
t95_sc = scipy_stats.t.ppf(0.975, df=n_sc - 2)
ax.fill_between(x_line_sc,
                y_line_sc - t95_sc * s_err_sc,
                y_line_sc + t95_sc * s_err_sc,
                color=u.ORANGE, alpha=0.08, zorder=2, label="95% Prediction Interval")

ax.scatter([cur_cape_sc], [cur_implied_sc], color=u.TEAL, s=200, zorder=6, marker="*",
           label=f"Today  CAPE={cur_cape_sc:.1f}  ->  {cur_implied_sc:.1f}%/yr")
ax.annotate(
    f"CAPE = {cur_cape_sc:.1f}\nImplied 10Y CAGR: {cur_implied_sc:.1f}%/yr",
    xy=(cur_cape_sc, cur_implied_sc),
    xytext=(18, -28), textcoords="offset points",
    color=u.TEAL, fontsize=9.5, fontweight="bold",
    arrowprops=dict(arrowstyle="-|>", color=u.TEAL, lw=1.2)
)

ax.axvspan(x_sc.min(), 15,            alpha=0.04, color=u.GREEN,  zorder=1)
ax.axvspan(15,          25,           alpha=0.03, color=u.YELLOW, zorder=1)
ax.axvspan(25,          35,           alpha=0.04, color=u.ORANGE, zorder=1)
ax.axvspan(35, x_sc.max()*1.05,       alpha=0.05, color=u.RED,    zorder=1)

decade_patches_sc = [
    mpatches.Patch(color=u.BLUE,   label="1990s"),
    mpatches.Patch(color=u.RED,    label="2000s"),
    mpatches.Patch(color=u.GREEN,  label="2010s"),
    mpatches.Patch(color=u.YELLOW, label="2020s"),
]
handles_sc, labels_sc = ax.get_legend_handles_labels()
ax.legend(handles_sc + decade_patches_sc,
          labels_sc + ["1990s", "2000s", "2010s", "2020s"],
          fontsize=8.5, facecolor=u.BORDER, edgecolor=u.BORDER,
          labelcolor=u.TEXT, loc="upper right")

ax.set_xlabel("CAPE (Shiller P/E)", color=u.TEXT, fontsize=11)
ax.set_ylabel("Annualized 10Y CAGR (%/yr)", color=u.TEXT, fontsize=11)
ax.text(0.02, 0.97,
        f"R² = {r2_sc:.3f}   (post-1990, n={len(x_sc)})\n"
        f"Each point = one month since 1990\n"
        f"Y = annualized CAGR over the following 10 years",
        transform=ax.transAxes, fontsize=8.5, color=u.DIM,
        va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=u.BORDER,
                  edgecolor=u.BORDER, alpha=0.8))

fig.suptitle("CAPE Implied 10Y Return",
             fontsize=13, color=u.TEXT, fontweight="bold", y=1.01)
u.save_fig(fig, "cape_scatter_10y.png")
