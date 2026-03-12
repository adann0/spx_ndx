"""eps.py - EPS -> output/eps.png"""
import sys
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

eps = u.load_multpl("spx_earning")
s = eps.sort_index().dropna()
s = s[s.index >= "1990-01-01"]

fig, ax = plt.subplots(figsize=(16, 6))
u.apply_style(fig, ax)
ax.plot(s.index, s.values, color=u.BLUE, lw=1.6, label="EPS ($)", zorder=3)
cur = s.iloc[-1]
ax.axhline(cur, color=u.BLUE, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax.set_ylabel("EPS ($)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax)
u.add_legend(ax)
u.add_stats(ax, s, label="EPS ($)")
fig.suptitle("EPS",
             fontsize=13, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "eps.png")
