"""credit_spreads.py - HY Credit Spreads -> output/credit_spreads.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

hyspr = u.load_fred("credit_spread")

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = hyspr.sort_index().dropna()
u.shade_recessions(ax)
ax.plot(s.index, s.values, color=u.YELLOW, lw=1.5, label="HY Credit Spread (%)", zorder=3)

cur = s.iloc[-1]
ax.axhline(cur, color=u.YELLOW, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("HY Spread (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="HY Credit Spread")
fig.suptitle("HY Credit Spreads",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "credit_spreads.png")
