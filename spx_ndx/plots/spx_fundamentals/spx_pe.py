"""spx_pe.py - SPX P/E Raw Full History -> output/spx_pe.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

pe = u.load_pq("spx_pe_ratio").clip(upper=60)

fig, ax = plt.subplots(figsize=(18, 6))
u.apply_style(fig, ax)

ax.axhspan(0,  15, alpha=0.07, color=u.GREEN,  zorder=0)
ax.axhspan(20, 28, alpha=0.07, color=u.ORANGE, zorder=0)
ax.axhspan(28, 65, alpha=0.07, color=u.RED,    zorder=0)

ax.plot(pe.index, pe.values, color=u.BLUE, lw=0.9, alpha=0.85,
        label="P/E TTM (capped 60)")
ax.axhline(pe.iloc[-1], color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")

ax.set_ylabel("P/E TTM", color=u.TEXT)
ax.set_ylim(0, 65)

u.fmt_xaxis(ax)
u.add_legend(ax, loc="upper left")
u.add_stats(ax, pe, label="P/E TTM")
fig.suptitle("P/E Ratio",
             fontsize=12, color=u.TEXT, fontweight="bold", y=1.01)
u.save_fig(fig, "spx_pe.png")
