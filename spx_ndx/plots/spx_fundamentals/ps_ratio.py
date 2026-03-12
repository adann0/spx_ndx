"""ps_ratio.py - P/S Ratio -> output/ps_ratio.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

ps = u.load_multpl("spx_price_to_sales_ratio")

s = ps.sort_index()

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)

ax.axhspan(0.0, 1.2, alpha=0.07, color=u.GREEN,  zorder=0)
ax.axhspan(2.0, 2.8, alpha=0.07, color=u.ORANGE, zorder=0)
ax.axhspan(2.8, 999, alpha=0.07, color=u.RED,    zorder=0)

ax.plot(s.index, s.values, color=u.BLUE, lw=1.5, zorder=3, label="P/S Ratio")
cur = s.iloc[-1]
ax.axhline(cur, color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")
pad = (s.max() - s.min()) * 0.1
ax.set_ylim(max(0, s.min() - pad), s.max() + pad)
ax.set_ylabel("P/S Ratio", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="P/S Ratio")
fig.suptitle("P/S Ratio",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "ps_ratio.png")
