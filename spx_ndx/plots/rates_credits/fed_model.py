"""fed_model.py - Earnings Yield vs 10Y (Fed Model) -> output/fed_model.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

earn_yld = u.load_multpl("spx_earnings_yield")
rate_10y = u.load_fred("rate_10y")

ey = earn_yld.resample("ME").last()
r10 = rate_10y.resample("ME").last()
ey, r10 = ey.align(r10, join="inner")
ey = ey[ey.index   >= "1990-01-01"]
r10 = r10[r10.index >= "1990-01-01"]
spread = ey - r10

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9),
                                gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})
u.apply_style(fig, [ax1, ax2])

ax1.plot(ey.index,  ey.values,  color=u.BLUE,   lw=1.5, label="Earnings Yield (1/P/E)")
ax1.plot(r10.index, r10.values, color=u.ORANGE, lw=1.5, label="10Y Treasury Rate")
ax1.fill_between(ey.index, ey.values, r10.values,
                 where=ey.values >= r10.values,
                 alpha=0.12, color=u.GREEN, label="Stocks attractive")
ax1.fill_between(ey.index, ey.values, r10.values,
                 where=ey.values < r10.values,
                 alpha=0.12, color=u.RED, label="Bonds attractive")
ax1.axhline(ey.iloc[-1],  color=u.BLUE,   lw=0.8, ls="--", alpha=0.7, label="Actual EY")
ax1.axhline(r10.iloc[-1], color=u.ORANGE, lw=0.8, ls="--", alpha=0.7, label="Actual 10Y")
ax1.set_ylabel("Yield (%)", color=u.TEXT, fontsize=10)
ax1.tick_params(labelbottom=False)
u.add_legend(ax1)

ax2.plot(spread.index, spread.values, color=u.TEAL, lw=1.3, label="Spread (EY - 10Y)")
ax2.axhline(0, color=u.DIM, lw=0.8, ls="--", alpha=0.6)
ax2.fill_between(spread.index, spread.values, 0,
                 where=spread.values >= 0, alpha=0.12, color=u.GREEN)
ax2.fill_between(spread.index, spread.values, 0,
                 where=spread.values < 0,  alpha=0.12, color=u.RED)
ax2.axhline(spread.iloc[-1], color=u.TEAL, lw=0.8, ls="--", alpha=0.7, label="Actual Spread")
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:+.1f}%"))
ax2.set_ylabel("Spread (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax2)
u.add_legend(ax2)
u.add_stats(ax2, spread, label="Spread")

fig.suptitle("Earnings Yield vs 10Y",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "fed_model.png")
