"""federal_debt.py - Federal Debt % of GDP -> output/federal_debt.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

fed_debt = u.load_fred("fed_debt_gdp")
s = fed_debt.sort_index().dropna()
s = s[s.index >= "1990-01-01"]
cur = s.iloc[-1]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(s.index, s.values, color=u.RED, lw=1.5, label="Federal Debt (% of GDP)", zorder=3)
ax.axhline(cur, color=u.RED,  lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
ax.set_ylabel("Federal Debt (% of GDP)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="Federal Debt (% of GDP)")
fig.suptitle("Federal Debt % of GDP",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "federal_debt.png")
