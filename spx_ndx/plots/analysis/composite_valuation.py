"""composite_valuation.py - SPX + Composite Valuation Index -> output/composite_valuation.png"""
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx_h = u.load_pq("spx_historical_prices")
cape_h = u.load_pq("spx_shiller_pe_ratio")
pe_h = u.load_pq("spx_pe_ratio").clip(upper=60)

cape_f, spx_f = cape_h.align(spx_h, join="inner")
pe_f, _ = pe_h.align(spx_h,   join="inner")
cape_z = (cape_f - cape_f.mean()) / cape_f.std()
pe_z = (pe_f   - pe_f.mean())   / pe_f.std()

fig, ax1 = plt.subplots(figsize=(18, 6))
u.apply_style(fig, ax1)
ax2 = u.add_twinx(ax1)

ax1.plot(spx_f.index, spx_f.values, color=u.RED, lw=0.9, label="SPX", zorder=3)
ax1.axhline(spx_f.iloc[-1], color=u.RED, lw=0.8, ls="--", alpha=0.7, label="Actual SPX")
ax1.set_yscale("log")
ax1.set_ylabel("SPX (log)", color=u.TEXT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

idx9 = cape_z.index.intersection(pe_z.index)
composite = ((cape_z.reindex(idx9) + pe_z.reindex(idx9)) / 2).dropna()
roll_2y = composite.rolling(24).mean()
ax2.plot(composite.index, composite.values, color=u.TEAL, lw=0.7, alpha=0.5,
         label="Composite (CAPE_z + P/E_z) / 2")
ax2.plot(roll_2y.index, roll_2y.values, color=u.TEAL, lw=1.8, label="2Y rolling mean")
ax2.axhline(0,  color=u.DIM,   lw=0.6, ls="--", alpha=0.4, label="z=0")
ax2.axhline(2,  color=u.RED,   lw=0.7, ls=":",  alpha=0.5, label="z=+2σ")
ax2.axhline(-1, color=u.GREEN, lw=0.7, ls=":",  alpha=0.4, label="z=-1σ")
ax2.axhline(roll_2y.iloc[-1], color=u.TEAL, lw=0.8, ls="--", alpha=0.7, label="Actual")
ax2.fill_between(composite.index, composite.values, 2,
                 where=composite.values >= 2, alpha=0.12, color=u.RED)
ax2.set_ylabel("Composite z-score (σ)", color=u.DIM)

u.merge_legends(ax1, ax2)
u.add_crashes_h(ax2)
u.fmt_xaxis(ax1)
u.add_stats(ax2, composite, label="Composite")
fig.suptitle("SPX & Composite Valuation",
             fontsize=12, color=u.TEXT, fontweight="bold", y=1.01)
u.save_fig(fig, "composite_valuation.png")
