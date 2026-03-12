"""real_rates.py - Real Interest Rates 10Y -> output/real_rates.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

real_r = u.load_fred("real_rate_10y")
s_full = real_r.sort_index().dropna()
s = s_full[s_full.index >= "1990-01-01"]
cur = s.iloc[-1]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(s.index, s.values, color=u.ORANGE, lw=1.5, label="Real Rate (%)", zorder=3)

ax.axhline(cur,  color=u.ORANGE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("Real Rate (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Real Rate (%)")
fig.suptitle("Real Interest Rates 10Y",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "real_rates.png")
