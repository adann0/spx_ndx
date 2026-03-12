"""corp_margins.py - Corporate Profit Margins -> output/corp_margins.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

gdp = u.load_fred("gdp")
corp_margins = u.load_fred("corp_margins")
gdp_annual = gdp.resample("YE").last()
corp_annual = corp_margins.resample("YE").last()
margins_pct = (corp_annual / gdp_annual * 100).dropna()

s = margins_pct.sort_index().dropna()
s = s[s.index >= "1990-01-01"]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(s.index, s.values, color=u.ORANGE, lw=1.6, label="Corp Margins (% of GDP)", zorder=3)
cur = s.iloc[-1]
ax.axhline(cur, color=u.ORANGE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.set_ylabel("Corp Margins (% of GDP)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Corp Margins (% of GDP)")
fig.suptitle("Corporate Profit Margins",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "corp_margins.png")
