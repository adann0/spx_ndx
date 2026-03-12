"""msci_ndx_spx.py - MSCI World vs NDX vs SPX -> output/msci_ndx_spx.png"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from spx_ndx import utils as u

spx = u.load_yahoo("gspc")
ndx = u.load_yahoo("ndx")
urth = u.load_yahoo("urth")
irx = u.load_yahoo("irx")

start = urth.index[0]
start_str = start.strftime("%b %Y")
spx_m = spx[spx.index   >= start]
ndx_m = ndx[ndx.index   >= start]
urth_m = urth[urth.index >= start]
irx_d = (irx[irx.index  >= start] / 100) / 252


def perf_metrics(s, rf, name):
    ret = s.pct_change().dropna()
    years = len(ret) / 252
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
    vol = ret.std() * np.sqrt(252)
    rf_a = rf.reindex(ret.index).ffill().fillna(0)
    sharpe = ((ret - rf_a).mean() * 252) / vol
    annual = (s.resample("YE").last().pct_change().dropna() * 100).round(1)

    def annualized(y):
        if len(s) < y * 252:
            return np.nan
        sub = s.iloc[-(y * 252):]
        return ((sub.iloc[-1] / sub.iloc[0]) ** (1 / y) - 1) * 100

    return {
        "name":   name,
        "cagr":   cagr * 100,
        "vol":    vol  * 100,
        "sharpe": sharpe,
        "annual": annual,
        "ann_1y":  annualized(1),
        "ann_5y":  annualized(5),
        "ann_10y": annualized(10),
    }


metrics = [
    perf_metrics(spx_m,  irx_d, "S&P 500"),
    perf_metrics(ndx_m,  irx_d, "Nasdaq-100"),
    perf_metrics(urth_m, irx_d, "MSCI World"),
]
bar_colors = [u.RED, u.BLUE, u.GREEN]

spx_rb2 = u.rebase(spx_m,  100)
ndx_rb2 = u.rebase(ndx_m,  100)
urth_rb2 = u.rebase(urth_m, 100)

fig = plt.figure(figsize=(20, 16))
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
u.apply_style(fig)

# Line chart (top, full width)
ax_l = fig.add_subplot(gs[0, :])
u.apply_style(fig, ax_l)
spx_pct = spx_rb2  - 100
ndx_pct = ndx_rb2  - 100
urth_pct = urth_rb2 - 100

ax_l.plot(spx_pct.index,  spx_pct.values,  color=u.RED,   lw=1.8, label="S&P 500")
ax_l.plot(ndx_pct.index,  ndx_pct.values,  color=u.BLUE,  lw=1.8, label="Nasdaq-100")
ax_l.plot(urth_pct.index, urth_pct.values, color=u.GREEN, lw=1.8, label="MSCI World")
ax_l.axhline(spx_pct.iloc[-1],  color=u.RED,   lw=0.8, ls="--", alpha=0.7)
ax_l.axhline(ndx_pct.iloc[-1],  color=u.BLUE,  lw=0.8, ls="--", alpha=0.7)
ax_l.axhline(urth_pct.iloc[-1], color=u.GREEN, lw=0.8, ls="--", alpha=0.7)
ax_l.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"+{x:.0f}%" if x >= 0 else f"{x:.0f}%"))
ax_l.axhline(0, color=u.DIM, lw=0.7, ls="--", alpha=0.4)
ax_l.set_ylabel("Return since inception (%)", color=u.TEXT, fontsize=10)
u.fmt_xaxis(ax_l)
u.add_legend(ax_l)
ax_l.set_title(f"Total Return since {start_str}",
               color=u.TEXT, fontsize=11, fontweight="bold")

# Table 1 - Key Metrics
ax_t1 = fig.add_subplot(gs[1, 0])
u.apply_style(fig, ax_t1)
ax_t1.axis("off")
tbl1_data = [
    [m["name"], f"{m['cagr']:.1f}%", f"{m['vol']:.1f}%", f"{m['sharpe']:.2f}"]
    for m in metrics
]
tbl1 = ax_t1.table(
    cellText=tbl1_data,
    colLabels=["Index", "CAGR", "Volatility", "Sharpe"],
    cellLoc="center", loc="center",
)
tbl1.auto_set_font_size(False)
tbl1.set_fontsize(13)
tbl1.scale(1, 3.2)
u.style_table(tbl1)
ax_t1.set_title(f"Key Metrics (since {start_str})",
                color=u.TEXT, fontsize=10, fontweight="bold", pad=12)

# Table 2 - Annualized returns
ax_t2 = fig.add_subplot(gs[1, 1])
u.apply_style(fig, ax_t2)
ax_t2.axis("off")
tbl2_data = [
    [m["name"],
     f"{m['ann_1y']:.1f}%"  if not np.isnan(m['ann_1y'])  else "N/A",
     f"{m['ann_5y']:.1f}%"  if not np.isnan(m['ann_5y'])  else "N/A",
     f"{m['ann_10y']:.1f}%" if not np.isnan(m['ann_10y']) else "N/A"]
    for m in metrics
]
tbl2 = ax_t2.table(
    cellText=tbl2_data,
    colLabels=["Index", "1Y", "5Y", "10Y"],
    cellLoc="center", loc="center",
)
tbl2.auto_set_font_size(False)
tbl2.set_fontsize(13)
tbl2.scale(1, 3.2)
u.style_table(tbl2)
ax_t2.set_title("Annualized Returns",
                color=u.TEXT, fontsize=10, fontweight="bold", pad=12)

# Annual returns bar chart
ax_b = fig.add_subplot(gs[2, :])
u.apply_style(fig, ax_b)
years = sorted(set.intersection(*[set(m["annual"].index) for m in metrics]))
x = np.arange(len(years))
w = 0.28
for i, m in enumerate(metrics):
    vals = [m["annual"].get(y, np.nan) for y in years]
    ax_b.bar(x + i*w, vals, w, label=m["name"], color=bar_colors[i], alpha=0.85)
ax_b.axhline(0, color=u.DIM, lw=0.8, ls="--", alpha=0.5)
ax_b.set_xticks(x + w)
ax_b.set_xticklabels([str(y.year) for y in years], rotation=45, ha="right", fontsize=8)
ax_b.set_ylabel("Annual Return (%)", color=u.TEXT, fontsize=9)
ax_b.set_title("Annual Returns by Year", color=u.TEXT, fontsize=10, fontweight="bold")
u.add_legend(ax_b)

fig.suptitle("MSCI World vs Nasdaq-100 vs S&P 500",
             fontsize=14, fontweight="bold", color=u.TEXT, y=1.01)
u.save_fig(fig, "msci_ndx_spx.png")
