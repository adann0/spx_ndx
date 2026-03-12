"""Export pipeline results to parquet and JSON."""

import json
from collections import Counter

import numpy as np
import pandas as pd


def _build_parquet(result, signal_names, tbill_returns):
    """Build the signal DataFrame for parquet export.

    Returns a DataFrame with signal, returns, agreement,
    plus per-signal raw columns.
    """
    oos_returns = result["oos_returns"]
    oos_signal = result["oos_signal"]
    oos_dates = result["oos_dates"]
    oos_agreement = result["oos_agreement"]
    oos_raw_signals = result["oos_raw_signals"]
    oos_buy_hold = result["oos_buy_hold"]

    oos_cash = tbill_returns.reindex(oos_dates).ffill().fillna(0).values
    df = pd.DataFrame({
        "signal": oos_signal, "strategy_returns": oos_returns, "buy_hold_returns": oos_buy_hold,
        "cash_returns": oos_cash, "agreement": oos_agreement,
    }, index=oos_dates)
    for j, signal_name in enumerate(signal_names):
        df[f"raw_{signal_name}"] = oos_raw_signals[:, j]
    return df


def _build_explain_json(result, signal_names):
    """Build the explainability dict for JSON export.

    Returns (explain_dict, global_percent).
    """
    oos_deltas = result["oos_deltas"]
    oos_raw_signals = result["oos_raw_signals"]
    fold_results = result["fold_results"]
    cagr_thresholds = result["cagr_thresholds"]

    n_signals = len(signal_names)

    # --- Signal usage ---
    signal_usage = Counter()
    for (min_votes, names), count in result["trader_counts"].items():
        for name in names:
            signal_usage[name] += count

    global_importance = np.mean(np.abs(oos_deltas), axis=0)
    total_global = global_importance.sum()
    global_percent = global_importance / total_global * 100 if total_global > 0 else np.zeros(n_signals)

    explain = {
        "signal_names": signal_names,
        "n_pipelines": len(cagr_thresholds),
    }

    structural_importance = result.get("last_structural_importance")
    if structural_importance is not None:
        explain["structural_importance"] = structural_importance.tolist()
        order = sorted(range(n_signals), key=lambda j: -structural_importance[j])
        explain["formula"] = [
            {"signal": signal_names[j], "weight": round(structural_importance[j] / 100, 4)}
            for j in order if structural_importance[j] >= 0.5
        ]
        last_raw = oos_raw_signals[-1]
        explain["current_formula_value"] = round(
            sum(structural_importance[j] / 100 * last_raw[j] for j in range(n_signals)), 4
        )

    explain["global_shapley_percent"] = global_percent.tolist()

    pnl_gain = result.get("pnl_gain")
    if pnl_gain is not None:
        explain["pnl_gain"] = result["pnl_gain"].tolist()
        explain["pnl_cost"] = result["pnl_cost"].tolist()
        explain["pnl_net"] = result["pnl_net"].tolist()

    explain["signal_usage"] = {name: signal_usage.get(name, 0) for name in signal_names}
    explain["fold_results"] = [
        {"period": r["period"], "test_cagr": r["test_cagr"], "buy_hold_cagr": r["buy_hold_cagr"]}
        for r in fold_results
    ]

    return explain, global_percent


def export_results(result, signal_names, tbill_returns):
    """Export signal parquet and explainability JSON.

    Returns (parquet_path, json_path, global_percent).
    """
    # --- Parquet ---
    parquet_path = "output/spx_consensus_signals.parquet"
    export_df = _build_parquet(result, signal_names, tbill_returns)
    export_df.to_parquet(parquet_path)

    # --- Explainability JSON ---
    json_path = "output/spx_consensus_explainability.json"
    explain, global_percent = _build_explain_json(result, signal_names)
    with open(json_path, "w") as f:
        json.dump(explain, f, indent=2)

    return parquet_path, json_path, global_percent
