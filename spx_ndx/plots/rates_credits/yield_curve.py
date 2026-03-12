"""yield_curve.py - Yield Curve 10Y−2Y -> output/yield_curve.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

yc = u.load_fred("yield_curve")

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = yc[yc.index >= "1990-01-01"].sort_index().dropna()
u.shade_recessions(ax)
ax.plot(s.index, s.values, color=u.BLUE, lw=1.4, label="10Y − 2Y Spread", zorder=3)
ax.fill_between(s.index, s.values, 0,
                where=s.values >= 0, alpha=0.08, color=u.GREEN)
ax.fill_between(s.index, s.values, 0,
                where=s.values < 0,  alpha=0.12, color=u.RED)
ax.axhline(0, color=u.DIM, lw=0.8, ls="--", alpha=0.5)
cur = s.iloc[-1]
ax.axhline(cur, color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.2f}%"))
ax.set_ylabel("Spread (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Yield Curve 10Y-2Y")
fig.suptitle("Yield Curve 10Y−2Y",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "yield_curve.png")
