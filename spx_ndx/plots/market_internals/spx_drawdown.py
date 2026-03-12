"""spx_drawdown.py - SPX Drawdown from ATH -> output/spx_drawdown.png"""
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx = u.load_yahoo("gspc")
spx_s = spx.sort_index()
dd = u.drawdown_from_ath(spx_s)
dd_s = dd[dd.index >= "1990-01-01"]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.fill_between(dd_s.index, dd_s.values, 0, alpha=0.4, color=u.RED, zorder=2)
ax.plot(dd_s.index, dd_s.values, color=u.RED, lw=1.0, zorder=3)

bears = [
    ("2002-10-09", -49.1, "Dot-com\n−49%"),
    ("2009-03-09", -56.8, "GFC\n−57%"),
    ("2020-03-23", -33.9, "COVID\n−34%"),
    ("2022-10-12", -25.4, "2022\n−25%"),
]
for date, val, label in bears:
    dt = pd.Timestamp(date)
    if dt in dd_s.index or dd_s.index.asof(dt) is not pd.NaT:
        ax.annotate(label,
                    xy=(dt, val),
                    xytext=(0, -28), textcoords="offset points",
                    color=u.DIM, fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="->", color=u.DIM, lw=0.6))

ax.axhline(0,   color=u.DIM,    lw=0.7, ls="--", alpha=0.4)
ax.axhline(-20, color=u.ORANGE, lw=0.8, ls=":", alpha=0.5,
           label="Bear market threshold (−20%)")
cur_dd = dd_s.iloc[-1]
ax.axhline(cur_dd, color=u.RED,  lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_ylabel("Drawdown from ATH (%)", color=u.TEXT, fontsize=10)
ax.set_ylim(dd_s.min() * 1.15, 5)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, dd_s, label="Drawdown")
fig.suptitle("SPX Drawdown from ATH",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "spx_drawdown.png")
