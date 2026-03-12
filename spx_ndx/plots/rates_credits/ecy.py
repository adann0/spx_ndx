"""ecy.py - Excess CAPE Yield (ECY) -> output/ecy.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

cape = u.load_multpl("spx_shiller_pe_ratio")
rate_10y = u.load_fred("rate_10y")

cape_yield = (1 / cape * 100)
rate_10y_s = rate_10y.sort_index()

breakeven = u.load_fred("breakeven_10y")
real_10y = (rate_10y_s.resample("ME").last() - breakeven.sort_index().resample("ME").last()).dropna()

cy_a, r10_a = cape_yield.align(real_10y, join="inner")
ecy = (cy_a - r10_a).sort_index().dropna()
cur = ecy.iloc[-1]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(ecy.index, ecy.values, color=u.TEAL, lw=1.5,
        label="Excess CAPE Yield (ECY)", zorder=3)
ax.fill_between(ecy.index, ecy.values, 0,
                where=ecy.values >= 0, alpha=0.08, color=u.GREEN,
                label="Stocks attractive")
ax.fill_between(ecy.index, ecy.values, 0,
                where=ecy.values < 0, alpha=0.08, color=u.RED,
                label="Bonds attractive")
ax.axhline(cur, color=u.TEAL, lw=0.8, ls="--", alpha=0.5, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
ax.set_ylabel("ECY (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, ecy, label="ECY")
ax.annotate("ECY = (1/CAPE) − Real 10Y Rate",
            xy=(0.01, 0.96), xycoords="axes fraction",
            fontsize=7.5, color=u.DIM, va="top")
fig.suptitle("Excess CAPE Yield (ECY)",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "ecy.png")
