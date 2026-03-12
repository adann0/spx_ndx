"""spx_cape_overlay.py - SPX (log) + CAPE overlay - Full History -> output/spx_cape_overlay.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx_h = u.load_pq("spx_historical_prices")
cape_h = u.load_pq("spx_shiller_pe_ratio")
cape_f, spx_f = cape_h.align(spx_h, join="inner")

fig, ax1 = plt.subplots(figsize=(18, 6))
u.apply_style(fig, ax1)
ax2 = u.add_twinx(ax1)

ax1.plot(spx_f.index, spx_f.values, color=u.RED, lw=0.9, label="SPX", zorder=3)
ax1.axhline(spx_f.iloc[-1], color=u.RED, lw=0.8, ls="--", alpha=0.7, label="Actual SPX")
ax1.set_yscale("log")
ax1.set_ylabel("SPX (log)", color=u.TEXT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

ax2.plot(cape_f.index, cape_f.values, color=u.PURPLE, lw=0.9, alpha=0.85, label="CAPE raw")
ax2.axhline(cape_f.iloc[-1], color=u.PURPLE, lw=0.8, ls="--", alpha=0.7, label="Actual")

ax2.set_ylabel("CAPE", color=u.DIM)

u.merge_legends(ax1, ax2)
u.add_crashes_h(ax2)
u.fmt_xaxis(ax1)
u.add_stats(ax2, cape_f, label="CAPE")
fig.suptitle("SPX & CAPE",
             fontsize=12, color=u.TEXT, fontweight="bold", y=1.01)
u.save_fig(fig, "spx_cape_overlay.png")
