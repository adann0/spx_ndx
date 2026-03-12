"""implied_return.py - CAPE-Implied 10Y CAGR over Time (OLS) -> output/implied_return.png"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats as scipy_stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx_h = u.load_pq("spx_historical_prices")
cape_h = u.load_pq("spx_shiller_pe_ratio")

HORIZON = 120  # 10 years in months
START_OLS = pd.Timestamp("1990-01-01")

# --- Fit OLS (identical to cape_scatter_10y.py) ---
cape_a, spx_a = cape_h.align(spx_h, join="inner")
fwd_raw = spx_a.shift(-HORIZON) / spx_a
fwd = (fwd_raw ** (1/10) - 1) * 100

common = cape_a.index.intersection(fwd.dropna().index)
cape_sel = cape_a.reindex(common)
fwd_sel = fwd.reindex(common)
mask = (np.isfinite(cape_sel.values) & np.isfinite(fwd_sel.values)
            & (common >= START_OLS))

slope, intercept, r, pval, _ = scipy_stats.linregress(
    cape_sel.values[mask], fwd_sel.values[mask])
r2 = r ** 2

print(f"  OLS  slope={slope:.3f}  intercept={intercept:.3f}  R²={r2:.3f}  p={pval:.2e}")
print(f"  Current CAPE: {cape_h.iloc[-1]:.1f}  ->  {slope * cape_h.iloc[-1] + intercept:.1f}%/yr")

# --- Apply OLS to all CAPE since 1990 -> implied time series ---
cape_ts = cape_h[cape_h.index >= "1990-01-01"].dropna()
implied = slope * cape_ts + intercept

# Realized 10Y CAGR where available (for visual validation)
fwd_realized = fwd[fwd.index >= "1990-01-01"].dropna()

cur = implied.iloc[-1]
cur_cape = cape_ts.iloc[-1]

# --- Plot ---
fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)

ax.plot(implied.index, implied.values, color=u.GREEN, lw=1.8,
        label=f"OLS implied CAGR 10Y  (slope={slope:.2f}, R²={r2:.2f})", zorder=4)

ax.plot(fwd_realized.index, fwd_realized.values, color=u.DIM, lw=1.0,
        ls="--", alpha=0.7, label="Realized 10Y CAGR", zorder=3)

ax.axhline(0,   color=u.DIM,    lw=0.7, ls="--", alpha=0.4)
ax.axhline(7.0, color=u.DIM,    lw=1.0, ls="--", alpha=0.6,
           label="Historical avg equity return (~7%)")
ax.axhline(cur, color=u.GREEN,  lw=0.8, ls="--", alpha=0.7,
           label=f"Actual  CAPE={cur_cape:.1f}  ->  {cur:.1f}%/yr")

ax.fill_between(implied.index, implied.values, 0,
                where=implied.values >= 0, alpha=0.07, color=u.GREEN)
ax.fill_between(implied.index, implied.values, 0,
                where=implied.values < 0,  alpha=0.07, color=u.RED)

ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("CAPE-implied 10Y CAGR (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, implied, label="Implied CAGR 10Y")

ax.annotate("Formula : CAGR 10Y = slope × CAPE + intercept  (OLS post-1990)",
            xy=(0.01, 0.96), xycoords="axes fraction",
            fontsize=7.5, color=u.DIM, va="top")

fig.suptitle("CAPE-Implied 10Y CAGR over Time",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "implied_return.png")
