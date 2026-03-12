"""Plot: PASS/FAIL summary table."""

import numpy as np
import matplotlib.pyplot as plt

from ._style import apply_style, save_fig, style_table, TEXT, DIM, BG, GRID, BORDER, GREEN, RED


def _compute_tests(metrics):
    """Compute all 13 PASS/FAIL results. Returns (rows, n_pass, n_total, verdict, base)."""
    base = metrics["baseline"]
    permutation_metrics = metrics["permutation"]
    vintage_metrics = metrics["vintage"]
    real_cagr = base["real_cagr"]
    bh_cagr = base["buy_hold_cagr"]
    real_max_drawdown = base["real_max_drawdown"]

    percent_beat_buy_hold = metrics["bootstrap"]["percent_beat_buy_hold"]
    block_bootstrap = metrics.get("block_bootstrap", {})
    correlation_bh = metrics.get("correlation_bh", {})

    perm_pass = permutation_metrics["p_cagr"] < 0.05
    boot_pass = percent_beat_buy_hold > 60
    noise_pass = (
        metrics["signal_noise"]["results"].get("20", {}).get("percent_beat", 0) > 50
    )
    vintage_pass = vintage_metrics["percent"] > 60
    bb_pass = block_bootstrap.get("rtr_mean", 0) > 0.5
    corr_pass = correlation_bh.get("corr_returns", 1.0) < 0.95

    txcosts_data = metrics.get("txcosts", {})
    txcosts_pass = False
    costs_bps = txcosts_data.get("costs_bps", [])
    adjusted_cagrs = txcosts_data.get("adjusted_cagrs", [])
    if 50 in costs_bps:
        txcosts_pass = adjusted_cagrs[costs_bps.index(50)] > bh_cagr

    retnoise_data = metrics.get("return_noise", {})
    retnoise_res = retnoise_data.get("results", {}).get("0.2", {})
    retnoise_cagrs = retnoise_res.get("cagrs", [])
    retnoise_pass = (sum(retnoise_cagrs) / len(retnoise_cagrs) > bh_cagr) if retnoise_cagrs else False

    decades_data = metrics.get("decades", {})
    decade_wins = sum(1 for d in decades_data.values()
                      if d["strategy_cagr"] > d["buy_hold_cagr"]) if decades_data else 0
    decade_total = len(decades_data)
    decade_pass = (decade_wins / decade_total > 0.5) if decade_total else False

    regimes_data = metrics.get("regimes", {})
    bear_regime = regimes_data.get("Bear (12M <= 0)", {})
    regime_pass = bear_regime.get("delta", 0) > 0

    bh_max_dd = base.get("buy_hold_max_drawdown", -1)
    dd_pass = abs(real_max_drawdown) < 0.5 * abs(bh_max_dd)

    alpha_pass = real_cagr > bh_cagr

    rolling_data = metrics.get("rolling_rtr") or {}
    rolling_sharpes = rolling_data.get("strategy_sharpe", [])
    if rolling_sharpes:
        pct_above_half = sum(1 for s in rolling_sharpes if s > 0.5) / len(rolling_sharpes) * 100
    else:
        pct_above_half = 0
    rolling_pass = pct_above_half > 80

    rows = [
        ("Permutation (p < 0.05)", perm_pass),
        ("Bootstrap (beats B&H > 60%)", boot_pass),
        ("Signal noise (holds at 20%)", noise_pass),
        ("Return noise (holds at 20%)", retnoise_pass),
        ("Txcosts (beats B&H at 50bps)", txcosts_pass),
        ("Vintage year (win > 60%)", vintage_pass),
        ("Decades (wins > 50%)", decade_pass),
        ("Regimes (bear alpha > 0)", regime_pass),
        ("Drawdowns (DD < 50% B&H)", dd_pass),
        ("Rolling Sharpe (> 0.5 80%)", rolling_pass),
        ("Block Bootstrap (RTR > 0.5)", bb_pass),
        ("Correlation B&H (r < 0.95)", corr_pass),
        ("Cumulative alpha (> 0)", alpha_pass),
    ]

    n_pass = sum(p for _, p in rows)
    n_total = len(rows)
    verdict = "OK" if n_pass >= n_total - 1 else "FAIL"

    return rows, n_pass, n_total, verdict, base


def plot_summary_table(metrics, df, label):
    """Full-width table, verdict integrated as last row, metrics as subtitle."""
    rows, n_pass, n_total, verdict, base = _compute_tests(metrics)

    real_cagr = base["real_cagr"]
    real_sharpe = base["real_sharpe"]
    real_max_drawdown = base["real_max_drawdown"]
    bh_cagr = base["buy_hold_cagr"]
    bh_sharpe = base["buy_hold_sharpe"]

    summary_data = [
        [str(i + 1), name, "PASS" if passed else "FAIL"]
        for i, (name, passed) in enumerate(rows)
    ]
    summary_data.append(["", f"VERDICT: {verdict}", f"{n_pass}/{n_total}"])

    fig, ax = plt.subplots(figsize=(8, 5.8))
    apply_style(fig, ax)
    ax.axis("off")

    table = ax.table(
        cellText=summary_data,
        colLabels=["#", "Test", "Verdict"],
        cellLoc="center", loc="upper center",
    )
    style_table(table)
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    table.auto_set_column_width([0, 1, 2])

    for j in range(3):
        table[0, j].set_text_props(fontweight="bold", fontsize=11)

    n_rows = len(summary_data)
    for i in range(1, n_rows + 1):
        table[i, 1].set_text_props(fontweight="bold")
        table[i, 1]._loc = "left"

        if i < n_rows:
            passed = rows[i - 1][1]
            cell = table[i, 2]
            if passed:
                cell.set_text_props(color=GREEN, fontweight="bold", fontsize=12)
                cell.set_facecolor("#0f1d0f")
            else:
                cell.set_text_props(color=RED, fontweight="bold", fontsize=12)
                cell.set_facecolor("#1d0f0f")
        else:
            verdict_color = GREEN if verdict == "OK" else RED
            for j in range(3):
                table[i, j].set_facecolor("#0f1d0f" if verdict == "OK" else "#1d0f0f")
                table[i, j].set_text_props(color=verdict_color, fontweight="bold",
                                           fontsize=13)

    fig.text(0.5, 0.98, "Stress Test Summary",
             ha="center", fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.5, 0.945,
             f"Strat: CAGR {real_cagr:+.1%}  Sharpe {real_sharpe:.2f}  "
             f"MaxDD {real_max_drawdown:.1%}   |   "
             f"B&H: CAGR {bh_cagr:+.1%}  Sharpe {bh_sharpe:.2f}  "
             f"MaxDD {base['buy_hold_max_drawdown']:.1%}",
             ha="center", fontsize=9, color=DIM)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.01)

    path = "output/spx_consensus_stress_summary.png"
    save_fig(fig, path)
    return fig, path
