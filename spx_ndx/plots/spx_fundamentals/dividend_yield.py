"""dividend_yield.py - Dividend Yield -> output/dividend_yield.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

div = u.load_multpl("spx_dividend_yield")

s_full = div.sort_index().copy()
s = s_full[s_full.index >= "1990-01-01"]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.axhspan(0,   1.5, alpha=0.07, color=u.RED,   zorder=0)
ax.axhspan(3.5, 999, alpha=0.07, color=u.GREEN,  zorder=0)
ax.plot(s.index, s.values, color=u.BLUE, lw=1.5, zorder=3, label="Dividend Yield (%)")
cur = s.iloc[-1]
ax.axhline(cur, color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")
pad = (s.max() - s.min()) * 0.1
ax.set_ylim(max(0, s.min() - pad), s.max() + pad)
ax.set_ylabel("Dividend Yield (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Dividend Yield")
fig.suptitle("Dividend Yield",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "dividend_yield.png")
