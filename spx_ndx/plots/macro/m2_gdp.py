"""m2_gdp.py - M2 / GDP -> output/m2_gdp.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

m2 = u.load_fred("m2")
gdp = u.load_fred("gdp")

m2_q = m2.resample("QE").last()
gdp_q = gdp.resample("QE").last().ffill()
m2_gdp = (m2_q / gdp_q * 100).sort_index().dropna()

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = m2_gdp[m2_gdp.index >= "1990-01-01"]
ax.plot(s.index, s.values, color=u.PURPLE, lw=1.5, label="M2 / GDP (%)", zorder=3)

cur = s.iloc[-1]
ax.axhline(cur, color=u.PURPLE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_ylabel("M2 / GDP (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="M2/GDP")
fig.suptitle("M2 / GDP",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "m2_gdp.png")
