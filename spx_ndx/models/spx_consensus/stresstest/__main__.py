"""CLI entry point for consensus stress tests.

Usage:
    python -m spx_ndx.models.spx_consensus.stresstest
    python -m spx_ndx.models.spx_consensus.stresstest output/spx_consensus_signals.parquet
    python -m spx_ndx.models.spx_consensus.stresstest --n 5000
"""

import argparse
import json

import numpy as np
import pandas as pd

from . import (
    StressData,
    compute_baseline,
    permutation_test,
    bootstrap_returns,
    txcosts_sensitivity,
    signal_noise_injection,
    return_noise_injection,
    montecarlo_paths,
    regime_split,
    vintage_analysis,
    drawdown_analysis,
    rolling_alpha,
    block_bootstrap,
    correlation_bh,
    baseline_signal_correlation,
    decade_split,
    rolling_rtr_sharpe,
)
from ._helpers import NumpyEncoder, load_stresstest_data, build_baseline_signals


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run consensus stress tests")
    parser.add_argument("parquet", nargs="?", default="output/spx_consensus_signals.parquet")
    parser.add_argument("--n", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default="datas/dataset_monthly.parquet",
                        help="Dataset parquet for cross-index and baseline signals")
    args = parser.parse_args(argv)

    df, signal, strategy_returns, buy_hold_returns, cash_returns, periods_per_year, label = load_stresstest_data(args.parquet)
    data = StressData(signal, strategy_returns, buy_hold_returns, cash_returns, periods_per_year)
    dates = df.index
    n_periods = len(signal)
    rng = np.random.default_rng(args.seed)

    print(f"Loaded {args.parquet}")
    print(f"  {n_periods} periods, ~{periods_per_year}/yr, {dates[0].date()} -> {dates[-1].date()}")
    print(f"  Exposure: {signal.mean():.0%}  ({int(signal.sum())}/{n_periods} periods invested)")

    # Baseline
    baseline = compute_baseline(data)
    print(f"\n  Strategy: CAGR={baseline['real_cagr']:+.1%}  RTR={baseline['real_rtr']:.2f}"
          f"  Sharpe={baseline['real_sharpe']:.2f}  MaxDD={baseline['real_max_drawdown']:.1%}")
    print(f"  B&H:      CAGR={baseline['buy_hold_cagr']:+.1%}  RTR={baseline['buy_hold_rtr']:.2f}"
          f"  Sharpe={baseline['buy_hold_sharpe']:.2f}")

    metrics = {
        "meta": {
            "label": label, "parquet": args.parquet, "frequency": "monthly",
            "n_periods": n_periods, "periods_per_year": periods_per_year,
            "n_iter": args.n, "seed": args.seed,
            "start": str(dates[0].date()), "end": str(dates[-1].date()),
        },
        "baseline": baseline,
    }

    n_iter = args.n

    # ── Significatif ? (1-2) ─────────────────────────────────────────────

    # TEST 1 - Permutation test
    _header(1, "Permutation test", n_iter)
    metrics["permutation"] = permutation_test(data, rng, n_iter, baseline)
    permutation_results = metrics["permutation"]
    print(f"  p(CAGR)={permutation_results['p_cagr']:.3f}  p(RTR)={permutation_results['p_rtr']:.3f}"
          f"  p(Sharpe)={permutation_results['p_sharpe']:.3f}  p(MaxDD)={permutation_results['p_max_drawdown']:.3f}")

    # TEST 2 - Bootstrap
    _header(2, "Bootstrap returns", n_iter)
    metrics["bootstrap"] = bootstrap_returns(data, rng, n_iter)
    bootstrap_results = metrics["bootstrap"]
    print(f"  CI CAGR: [{bootstrap_results['ci_cagr'][0]:+.1%}, {bootstrap_results['ci_cagr'][1]:+.1%}]"
          f"  Beat B&H: {bootstrap_results['percent_beat_buy_hold']:.0f}%")

    # ── Robuste ? (3-5) ───────────────────────────────────────────────

    # TEST 3 - Signal noise
    _header(3, "Signal noise injection")
    metrics["signal_noise"] = signal_noise_injection(data, rng, n_iter, baseline)

    # TEST 4 - Return noise
    _header(4, "Return noise injection")
    metrics["return_noise"] = return_noise_injection(data, rng, n_iter)

    # TEST 5 - Transaction costs
    _header(5, "Transaction costs sensitivity")
    metrics["txcosts"] = txcosts_sensitivity(data, baseline)
    txcosts_results = metrics["txcosts"]
    breakeven = f"{txcosts_results['breakeven']:.0f}bps" if txcosts_results["breakeven"] else "N/A"
    print(f"  Trades: {txcosts_results['n_trades']}  Breakeven: {breakeven}")

    # ── Stable dans le temps ? (6-10) ─────────────────────────────────

    # TEST 6 - Vintage analysis
    _header(6, "Vintage analysis")
    metrics["vintage"] = vintage_analysis(data, df)
    vintage_results = metrics["vintage"]
    print(f"  Wins: {vintage_results['wins']}/{vintage_results['total']} ({vintage_results['percent']:.0f}%)")

    # TEST 7 - Decade split
    _header(7, "Decade split")
    metrics["decades"] = decade_split(data, dates)

    # TEST 8 - Regime split
    _header(8, "Regime split")
    try:
        dataset = pd.read_parquet(args.dataset)
        dataset.index = pd.to_datetime(dataset.index)
        vix = dataset["vix_close"].reindex(dates).ffill()
    except Exception:
        vix = pd.Series(np.full(n_periods, 20.0), index=dates)
    metrics["regimes"] = regime_split(data, dates, vix)

    # TEST 9 - Drawdowns
    _header(9, "Drawdown analysis")
    metrics["drawdowns"] = drawdown_analysis(data, dates)

    # TEST 10 - Rolling Sharpe
    _header(10, "Rolling Sharpe")
    metrics["rolling_rtr"] = rolling_rtr_sharpe(data, dates)

    # ── Tests complementaires (11-15) ─────────────────────────────────

    # TEST 11 - Block bootstrap
    _header(11, "Block bootstrap", n_iter)
    metrics["block_bootstrap"] = block_bootstrap(data, rng, n_iter)

    # TEST 12 - Correlation to B&H
    _header(12, "Correlation to B&H")
    metrics["correlation_bh"] = correlation_bh(data)
    correlation_results = metrics["correlation_bh"]
    print(f"  corr(returns)={correlation_results['corr_returns']:.3f}  corr(|ret|)={correlation_results['corr_abs']:.3f}")

    # TEST 13 - Monte Carlo
    _header(13, "Monte Carlo paths", n_iter)
    metrics["montecarlo"] = montecarlo_paths(data, rng, n_iter)
    montecarlo_results = metrics["montecarlo"]
    print(f"  Beat B&H: {montecarlo_results['montecarlo_beat']:.0f}%  Alpha: {montecarlo_results['montecarlo_alpha']:+.2%}")

    # TEST 14 - Rolling alpha
    _header(14, "Rolling alpha")
    metrics["rolling_alpha"] = rolling_alpha(data, dates)

    # TEST 15 - Baseline signal correlation
    _header(15, "Baseline signal correlation")
    try:
        baseline_sigs = build_baseline_signals(args.dataset, dates)
    except Exception:
        baseline_sigs = {}
    if baseline_sigs:
        trend_names = ["SMA 10m", "12M mom", "EMA200"]
        trend_only = {k: v for k, v in baseline_sigs.items() if k in trend_names}
        base_only = {k: v for k, v in baseline_sigs.items() if k not in trend_names}
        metrics["trend_correlation"] = baseline_signal_correlation(signal, trend_only)
        metrics["baseline_correlation"] = baseline_signal_correlation(signal, base_only)
        for name, arr in baseline_sigs.items():
            corr = float(np.corrcoef(signal, arr)[0, 1])
            print(f"  corr(consensus, {name:>12s}) = {corr:.3f}")
    else:
        metrics["trend_correlation"] = {"correlations": {}, "matrix": [], "labels": []}
        metrics["baseline_correlation"] = {"correlations": {}, "matrix": [], "labels": []}

    # Last 12 periods
    last12 = df.tail(12)
    metrics["last_12"] = {
        "dates": last12.index.strftime("%Y-%m").tolist(),
        "signals": last12["signal"].values.tolist(),
        "strategy_returns": last12["strategy_returns"].values.tolist(),
        "buy_hold_returns": last12["buy_hold_returns"].values.tolist(),
    }

    # Export
    out_path = "output/spx_consensus_stress_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    print(f"\nExported metrics to {out_path}")

    return metrics


def _header(num, title, n_iterations=None):
    suffix = f" ({n_iterations} iter)" if n_iterations else ""
    print(f"\n{'='*70}")
    print(f"  {num}. {title}{suffix}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
