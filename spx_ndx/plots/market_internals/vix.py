"""vix.py - VIX Volatility Index -> output/vix.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

vix = u.load_yahoo("vix")
vix_s = vix.sort_index()
s = vix_s[vix_s.index >= "1990-01-01"]
cur = s.iloc[-1]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.axhspan(0,  20, alpha=0.07, color=u.GREEN,  zorder=0)
ax.axhspan(20, 30, alpha=0.07, color=u.ORANGE, zorder=0)
ax.axhspan(30, 80, alpha=0.07, color=u.RED,    zorder=0)
ax.plot(s.index, s.values, color=u.BLUE, lw=1.2, label="VIX", zorder=3)

ax.axhline(cur,  color=u.BLUE,   lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.set_ylim(0, s.max() * 1.05)
ax.set_ylabel("VIX", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="VIX")
fig.suptitle("VIX",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "vix.png")
