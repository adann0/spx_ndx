"""cpi_inflation.py - CPI Inflation Rate YoY -> output/cpi_inflation.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

cpi = u.load_fred("cpi")
cpi_yoy = cpi.sort_index().ffill().pct_change(12, fill_method=None) * 100
cpi_yoy = cpi_yoy.dropna()

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
s = cpi_yoy[cpi_yoy.index >= "1990-01-01"]
u.shade_recessions(ax)
ax.plot(s.index, s.values, color=u.RED, lw=1.5, label="CPI YoY (%)", zorder=3)
ax.axhline(2.0, color=u.DIM,    lw=1.0, ls="--", alpha=0.6, label="Fed Target (2%)")
ax.axhline(0.0, color=u.DIM, lw=0.7, ls=":", alpha=0.5)
cur = s.iloc[-1]
ax.axhline(cur, color=u.RED,  lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("CPI YoY (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="CPI YoY")
fig.suptitle("CPI Inflation Rate",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "cpi_inflation.png")
