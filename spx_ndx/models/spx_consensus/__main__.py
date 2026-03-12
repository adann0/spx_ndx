"""CLI entry point: python -m spx_ndx.models.spx_consensus [config.yaml]"""

import sys

import numpy as np
import pandas as pd

from spx_ndx.models.spx_consensus.config import load_config, FREQ_PERIODS
from spx_ndx.models.spx_consensus.signals import compute_signals
from spx_ndx.models.spx_consensus.grid import make_configs, warmup
from spx_ndx.models.spx_consensus.pipeline import run_pipeline
from spx_ndx.models.spx_consensus.export import export_results


_SEPARATOR = "\u2550" * 70
_DOT_SEPARATOR = " \u00b7" * 35
_GREEN, _RED, _ORANGE, _BOLD, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[1m", "\033[0m"


def _format_transitions(transitions):
    """Format transition list into a printable string."""
    if not transitions:
        return "\n  No transitions"
    lines = [f"\n  Transitions ({len(transitions)})"]
    for transition_index, transition in enumerate(transitions):
        n_valid = transition["n_valid"]
        decision = "IN" if transition["direction"] == "OUT->IN" else "OUT"
        agreement_count = transition["agreement"] if decision == "IN" else n_valid - transition["agreement"]
        changed_str = "  ".join(transition["changed"][:6])
        if len(transition["changed"]) > 6:
            changed_str += f"  +{len(transition['changed'])-6}"
        drivers_str = ", ".join(transition["drivers"][:3]) if transition["drivers"] else "(unanimous)"
        votes = transition["votes"]
        votes_str = "  ".join(f"{v:.2f}" for v in votes if not np.isnan(v))
        lines.append(f"    {transition['date'].strftime('%Y-%m')}  {decision}")
        lines.append(f"      agreement  {agreement_count}/{n_valid}")
        lines.append(f"      votes      {votes_str}")
        lines.append(f"      signals    {changed_str}")
        lines.append(f"      drivers    {drivers_str}")
        if transition_index < len(transitions) - 1:
            lines.append("")
    return "\n".join(lines)


def _print_fold_detail(fold_result, signal_names):
    """Print detailed results for a single fold."""
    print(f"\n{_SEPARATOR}")
    print(f"Fold {fold_result['fold']} \u2502 train {fold_result['train_start']}\u2013{fold_result['train_end']} \u2502 test {fold_result['period']}")
    print(_SEPARATOR)

    # HP sweep table
    sweep = fold_result.get("sweep_data")
    if sweep and sweep["all_results"]:
        hp_names = sweep["hp_names"]
        header_line = "  ".join(f"{k:>16s}" for k in hp_names)
        print(f"\n    {header_line}  {'cagr':>7s}  {'rtr':>7s}  {'sharpe':>7s}  {'stability':>9s}  {'ens':>3s}")
        floor_metrics = sweep["floor_metrics"]
        if floor_metrics:
            floor_values = "  ".join(f"{sweep['base_values'][k]!s:>16s}" for k in hp_names)
            print(f"    {floor_values}  {floor_metrics['cagr']:>7.1%}  {floor_metrics['rtr']:>7.3f}  {floor_metrics.get('sharpe', 0):>7.3f}"
                  f"  {floor_metrics['stability']:>8.4f}  {sweep['base_n_ensembles']:>3d}   baseline")
        for score, sweep_metrics, combo, overrides, passes, _, n_ensembles in sorted(
            sweep["all_results"], key=lambda x: (bool(x[4]), x[0]), reverse=True
        ):
            combo_values = "  ".join(f"{v!s:>16s}" for v in combo)
            suffix = " *" if overrides == sweep["best_combo"] else ("   f" if not passes else "")
            print(f"    {combo_values}  {sweep_metrics['cagr']:>7.1%}  {sweep_metrics['rtr']:>7.3f}  {sweep_metrics.get('sharpe', 0):>7.3f}"
                  f"  {sweep_metrics['stability']:>8.4f}  {n_ensembles:>3d}{suffix}")

    if fold_result.get("all_failed"):
        print(f"  All thresholds failed -> B&H")
        print(f"\n  ({fold_result['duration']:.1f}s)")
        return

    # Train/test metrics
    metrics_train = fold_result["metrics_train"]
    metrics_buy_hold = fold_result["metrics_buy_hold_train"]
    print(f"\n{_DOT_SEPARATOR}")
    print(f"\n                CAGR     RTR  Sharpe   MaxDD   Exp")
    print(f"  Train  Strat {metrics_train['cagr']:>5.1%} {metrics_train['rtr']:>7.2f} {metrics_train['sharpe']:>7.2f} {metrics_train['max_drawdown']:>7.1%} {fold_result['train_exposure']:>5.0%}")
    print(f"         B&H   {metrics_buy_hold['cagr']:>5.1%} {metrics_buy_hold['rtr']:>7.2f} {metrics_buy_hold['sharpe']:>7.2f} {metrics_buy_hold['max_drawdown']:>7.1%}")
    print(f"  Test   Strat {fold_result['test_cagr']:>5.1%} {fold_result['test_rtr']:>7.2f} {fold_result['test_sharpe']:>7.2f} {fold_result['test_max_drawdown']:>7.1%} {fold_result['test_exposure']:>5.0%}")
    print(f"         B&H   {fold_result['buy_hold_cagr']:>5.1%} {fold_result['buy_hold_rtr']:>7.2f} {fold_result['buy_hold_sharpe']:>7.2f} {fold_result['buy_hold_max_drawdown']:>7.1%}")

    # Transitions
    transitions = fold_result.get("transitions", [])
    print(f"\n{_DOT_SEPARATOR}")
    print(_format_transitions(transitions))

    # Top signals
    structural_importance = fold_result["structural_importance"]
    percent_delta = fold_result["percent_delta"]
    print(f"\n{_DOT_SEPARATOR}")
    print(f"\n  Top signals")
    print(f"  {'':18s}{'structural':>10s}{'Shapley':>10s}")
    combined = np.maximum(structural_importance, percent_delta)
    order_top = np.argsort(-combined)
    for j in order_top:
        if structural_importance[j] < 1.0 and percent_delta[j] < 1.0:
            continue
        print(f"    {signal_names[j]:16s}{structural_importance[j]:>9.0f}%{percent_delta[j]:>9.0f}%")

    print(f"\n  ({fold_result['duration']:.1f}s)")


def _format_fold_row(fold_result, hp_keys, hp_widths, fmt_hp):
    """Format one fold row for the summary table. Returns a string."""
    strategy_line = (f"{fold_result['test_rtr']:>5.2f}  {fold_result['test_sharpe']:>5.2f}  "
                     f"{fold_result['test_cagr']:>6.1%}  {fold_result['test_max_drawdown']:>6.1%}")
    buy_hold_line = (f"{fold_result['buy_hold_rtr']:>5.2f}  {fold_result['buy_hold_sharpe']:>5.2f}  "
                     f"{fold_result['buy_hold_cagr']:>6.1%}  {fold_result['buy_hold_max_drawdown']:>6.1%}")
    hyperparameters = fold_result.get("adapted_hps", {})
    if hp_keys:
        hyperparameter_values = "  ".join(fmt_hp(hyperparameters.get(k, ""), w)
                                          for k, w in zip(hp_keys, hp_widths))
        return (f"  {fold_result['fold']:>4}  {fold_result['period']:>10}  \u2502  {hyperparameter_values}  "
                f"\u2502  {strategy_line}  \u2502  {buy_hold_line}")
    return f"  {fold_result['fold']:>4}  {fold_result['period']:>10}  \u2502  {strategy_line}  \u2502  {buy_hold_line}"


def _print_summary(fold_results, metrics_oos, metrics_buy_hold_oos, oos_signal, total_duration, config,
                   signal_names=None, pnl_gain=None, pnl_cost=None, pnl_net=None):
    """Print the OOS summary table."""
    print(f"\n{_SEPARATOR}")
    print(f"RESULTS  [{total_duration:.0f}s total]")
    print(f"{_SEPARATOR}")

    # PnL attribution
    if pnl_net is not None:
        order_pnl = np.argsort(pnl_net)
        print(f"\n  PnL attribution (cumulative pp)")
        print(f"  {'':18s}{'gain':>8s}{'cost':>8s}{'net':>8s}")
        for j in order_pnl:
            if abs(pnl_net[j]) < 0.1:
                continue
            color = _GREEN if pnl_net[j] >= 0 else _RED
            print(f"    {signal_names[j]:16s}{pnl_gain[j]:>+7.1f}{pnl_cost[j]:>+7.1f}  {color}{_BOLD}{pnl_net[j]:>+6.1f}{_RESET}")
        print(f"\n{_DOT_SEPARATOR}")

    print(f"\n  Walk-forward folds")

    adaptive_grid = config.adaptive_grid
    _hp_short_names = {"vote_threshold": "vt", "trader_min_rtr": "t_rtr"}
    _hp_key_names = list(adaptive_grid.keys()) if adaptive_grid else []
    _hp_label_names = [_hp_short_names.get(k, k[:6]) for k in _hp_key_names]

    def _format_hyperparameter_raw(val):
        if isinstance(val, tuple):
            return ",".join(str(s)[:4] for s in val)
        if isinstance(val, float):
            return f"{val:g}"
        return str(val)

    _hp_column_widths = []
    for k, label in zip(_hp_key_names, _hp_label_names):
        max_val_w = max((len(_format_hyperparameter_raw(v)) for v in adaptive_grid[k]), default=4)
        _hp_column_widths.append(max(len(label), max_val_w))

    def _format_hyperparameter(val, w):
        return f"{_format_hyperparameter_raw(val):>{w}}"

    _hp_header_line = "  ".join(f"{label:>{w}}" for label, w in zip(_hp_label_names, _hp_column_widths))
    _hp_separator_width = len(_hp_header_line) if _hp_header_line else 0

    if _hp_key_names:
        print(f"\n  {'Fold':>4}  {'Period':>10}  \u2502  {'HPs':^{_hp_separator_width}}  \u2502  {'Consensus':^40}  \u2502  {'B&H':^40}")
        print(f"  {'':>16}  \u2502  {_hp_header_line}  \u2502  {'RTR':>5}  {'Shrp':>5}  {'CAGR':>6}  {'MaxDD':>6}  \u2502  {'RTR':>5}  {'Shrp':>5}  {'CAGR':>6}  {'MaxDD':>6}")
        print(f"  {'-'*16}--{'-'*(_hp_separator_width+2)}--{'-'*42}--{'-'*42}")
    else:
        print(f"\n  {'Fold':>4}  {'Period':>10}  \u2502  {'Consensus':^40}  \u2502  {'B&H':^40}")
        print(f"  {'':>16}  \u2502  {'RTR':>5}  {'Shrp':>5}  {'CAGR':>6}  {'MaxDD':>6}  \u2502  {'RTR':>5}  {'Shrp':>5}  {'CAGR':>6}  {'MaxDD':>6}")
        print(f"  {'-'*16}--{'-'*42}--{'-'*42}")

    for fold_result in fold_results:
        print(_format_fold_row(fold_result, _hp_key_names, _hp_column_widths, _format_hyperparameter))

    # Final metrics
    _cagr_color = _GREEN if metrics_oos["cagr"] > metrics_buy_hold_oos["cagr"] else _RED
    _rtr_color = _GREEN if metrics_oos["rtr"] > metrics_buy_hold_oos["rtr"] else _RED
    _sharpe_color = _GREEN if metrics_oos["sharpe"] > metrics_buy_hold_oos["sharpe"] else _RED
    _drawdown_color = _GREEN if metrics_oos["max_drawdown"] < metrics_buy_hold_oos["max_drawdown"] else _RED
    print(f"\n  {_BOLD}Strategy{_RESET}: {_BOLD}CAGR{_RESET}: {_BOLD}{_cagr_color}{metrics_oos['cagr']:.1%}{_RESET}  "
          f"{_BOLD}MaxDD{_RESET}: {_BOLD}{_drawdown_color}{metrics_oos['max_drawdown']:.1%}{_RESET}  "
          f"{_BOLD}RTR{_RESET}: {_BOLD}{_rtr_color}{metrics_oos['rtr']:.2f}{_RESET}  "
          f"{_BOLD}Sharpe{_RESET}: {_BOLD}{_sharpe_color}{metrics_oos['sharpe']:.2f}{_RESET}  "
          f"{_BOLD}Exposure{_RESET}: {oos_signal.mean():.0%}")
    print(f"  {_BOLD}B&H{_RESET}:      {_BOLD}CAGR{_RESET}: {_BOLD}{_ORANGE}{metrics_buy_hold_oos['cagr']:.1%}{_RESET}  "
          f"{_BOLD}MaxDD{_RESET}: {_BOLD}{_ORANGE}{metrics_buy_hold_oos['max_drawdown']:.1%}{_RESET}  "
          f"{_BOLD}RTR{_RESET}: {_BOLD}{_ORANGE}{metrics_buy_hold_oos['rtr']:.2f}{_RESET}  "
          f"{_BOLD}Sharpe{_RESET}: {_BOLD}{_ORANGE}{metrics_buy_hold_oos['sharpe']:.2f}{_RESET}")


def main(config_path: str = "spx_consensus.yaml"):
    # --- Load config ---
    config = load_config(config_path)
    periods_per_year = float(FREQ_PERIODS[config.frequency])

    # --- Load dataset ---
    dataset = pd.read_parquet(f"datas/dataset_{config.frequency}.parquet")
    dataset.index = pd.to_datetime(dataset.index)
    closes = dataset["spx_close"]
    returns = closes.pct_change().dropna()
    tbill_returns = (1 + dataset["tbill_rate"] / 100) ** (1 / periods_per_year) - 1

    # --- Compute signals ---
    signals_dict = compute_signals(dataset, closes, config.indicators)
    signals = {k: v for k, v in signals_dict.items() if not v.isna().all()}
    signal_names = list(signals.keys())
    n_signals = len(signal_names)

    # --- JIT warmup ---
    warmup()

    # --- Header ---
    _n_trader_configs = len(make_configs(n_signals, config.min_signals_per_trader, config.max_signals_per_trader))
    _cagr_lowest = int(min(config.cagr_thresholds) * 100)
    _cagr_highest = int(max(config.cagr_thresholds) * 100)

    print(f"\n  {'frequency':11s}{config.frequency}")
    print(f"  {'dataset':11s}{dataset.shape[0]} rows, {dataset.index[0].strftime('%Y-%m')} to {dataset.index[-1].strftime('%Y-%m')}")
    print(f"  {'signals':11s}{n_signals}")
    print(f"  {'traders':11s}{_n_trader_configs:,}")
    print(f"  {'ensemble':11s}{len(config.cagr_thresholds)} thresholds [{_cagr_lowest}-{_cagr_highest}%], majority ceil(n/2), expanding")

    # --- Run pipeline ---
    result = run_pipeline(config, signals, signal_names, returns, tbill_returns, periods_per_year, dataset.index[-1].year)

    if result is None:
        print("No results")
        raise SystemExit(1)

    # --- Per-fold details ---
    fold_results = result["fold_results"]
    for fold_result in fold_results:
        _print_fold_detail(fold_result, signal_names)

    # --- Summary ---
    _print_summary(fold_results, result["metrics_oos"], result["metrics_buy_hold_oos"],
                   result["oos_signal"], result["total_duration"], config,
                   signal_names, result["pnl_gain"], result["pnl_cost"], result["pnl_net"])

    # --- Export ---
    parquet_path, json_path, _ = export_results(result, signal_names, tbill_returns)
    print(f"\n  Exported {parquet_path}")
    print(f"  Exported {json_path}")


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "spx_consensus.yaml"
    main(config_file)
