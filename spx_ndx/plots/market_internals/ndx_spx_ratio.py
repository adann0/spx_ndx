"""ndx_spx_ratio.py - NDX/SPX Ratio (Tech Dominance) -> output/ndx_spx_ratio.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

ndx = u.load_yahoo("ndx")
spx = u.load_yahoo("gspc")

ndx_s, spx_s = ndx.sort_index().align(spx.sort_index(), join="inner")
ratio = (ndx_s / spx_s)
ratio_s = ratio[ratio.index >= "1990-01-01"].dropna()
cur = ratio_s.iloc[-1]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(ratio_s.index, ratio_s.values, color=u.BLUE, lw=1.5,
        label="NDX / SPX Ratio", zorder=3)
ax.axhline(cur,  color=u.BLUE,   lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.set_ylabel("NDX / SPX", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, ratio_s, label="NDX/SPX ratio")
fig.suptitle("NDX / SPX Ratio - Tech Dominance",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "ndx_spx_ratio.png")
