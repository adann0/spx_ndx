"""unemployment.py - US Unemployment Rate -> output/unemployment.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

unemp = u.load_fred("unemployment")

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = unemp[unemp.index >= "1990-01-01"].sort_index().dropna()
u.shade_recessions(ax)
ax.plot(s.index, s.values, color=u.ORANGE, lw=1.5, label="Unemployment Rate (%)", zorder=3)

cur = s.iloc[-1]
ax.axhline(cur, color=u.ORANGE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("Unemployment Rate (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Unemployment")
fig.suptitle("Unemployment",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "unemployment.png")
