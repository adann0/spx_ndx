"""buffett_indicator.py - Buffett Indicator -> output/buffett_indicator.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

w5000 = u.load_yahoo("w5000")
gdp = u.load_fred("gdp")

w5000_q = w5000.resample("QE").last()
gdp_q = gdp.resample("QE").last().ffill()
buffett = (w5000_q / gdp_q * 100).sort_index().dropna()

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = buffett[buffett.index >= "1990-01-01"]
ax.axhspan(0,   100, alpha=0.07, color=u.GREEN,  zorder=0)
ax.axhspan(100, 200, alpha=0.07, color=u.ORANGE, zorder=0)
ax.axhspan(200, 250, alpha=0.07, color=u.RED,    zorder=0)
ax.set_ylim(0, 250)
ax.plot(s.index, s.values, color=u.BLUE, lw=1.5, label="Wilshire 5000 / GDP (%)", zorder=3)

cur = s.iloc[-1]
ax.axhline(cur, color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_ylabel("Wilshire 5000 / GDP (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Buffett Indicator")
fig.suptitle("Buffett Indicator",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "buffett_indicator.png")
