"""spx_cape_pe_spread.py - SPX + Spread CAPE_z - PE_z -> output/spx_cape_pe_spread.png"""
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
pe_f,   _ = pe_h.align(spx_h,   join="inner")
cape_z = (cape_f - cape_f.mean()) / cape_f.std()
pe_z = (pe_f   - pe_f.mean())   / pe_f.std()
spread = (cape_z - pe_z).dropna()

fig, ax1 = plt.subplots(figsize=(18, 6))
u.apply_style(fig, ax1)
ax2 = u.add_twinx(ax1)

ax1.plot(spx_f.index, spx_f.values, color=u.RED, lw=0.9, label="SPX", zorder=3)
ax1.axhline(spx_f.iloc[-1], color=u.RED, lw=0.8, ls="--", alpha=0.7, label="Actual SPX")
ax1.set_yscale("log")
ax1.set_ylabel("SPX (log)", color=u.TEXT)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

sp = spread.dropna()
ax2.plot(sp.index, sp.values, color=u.YELLOW, lw=0.9, alpha=0.9, label="CAPE_z − P/E_z")
ax2.axhline(0,          color=u.DIM,    lw=0.7, ls="--", alpha=0.5, label="zero (CAPE=P/E)")
ax2.axhline(sp.iloc[-1], color=u.YELLOW, lw=0.8, ls="--", alpha=0.7, label="Actual")

ax2.fill_between(sp.index, sp.values, 0, where=sp.values > 0, alpha=0.08, color=u.RED,
                 label="CAPE > P/E")
ax2.fill_between(sp.index, sp.values, 0, where=sp.values < 0, alpha=0.08, color=u.GREEN,
                 label="P/E > CAPE")
sp_abs_max = max(abs(sp.quantile(0.02)), abs(sp.quantile(0.98))) * 1.1
ax2.set_ylim(-sp_abs_max, sp_abs_max)
ax2.set_ylabel("CAPE_z − P/E_z (σ)", color=u.DIM)

u.merge_legends(ax1, ax2)
u.add_crashes_h(ax2)
u.fmt_xaxis(ax1)
u.add_stats(ax2, sp, label="Spread")
fig.suptitle("SPX & Spread",
             fontsize=12, color=u.TEXT, fontweight="bold", y=1.01)
u.save_fig(fig, "spx_cape_pe_spread.png")
