"""Walk-forward consensus pipeline - core logic."""

import time
from collections import Counter
from dataclasses import dataclass, replace
from itertools import product as iterproduct

import numpy as np
import pandas as pd

from spx_ndx.models.spx_consensus.combo import combo_signal
from spx_ndx.models.spx_consensus.evaluate import (
    compute_strategy_returns, compute_all_metrics,
)
from spx_ndx.models.spx_consensus.explain import (
    compute_ensemble_vote, signal_importance_ensemble,
    structural_importance_ensemble, signal_pnl_attribution,
)
from spx_ndx.models.spx_consensus.grid import make_configs, _eval_all, grid_search
from spx_ndx.models.spx_consensus.signals import build_sig_matrix


# Pipeline context

@dataclass(slots=True)
class SplitData:
    """Signal matrix, returns, and cash for a train/val/test split."""
    signal_matrix: np.ndarray
    returns: np.ndarray
    cash_returns: np.ndarray


@dataclass
class _Context:
    """Shared context across folds (immutable)."""
    config: object
    signals: dict
    signal_names: list
    n_signals: int
    returns: object
    tbill_returns: object
    trader_configs: np.ndarray
    periods_per_year: float


# Helpers

def _group_weights(top_groups, aggregation):
    """Compute group weights based on aggregation mode."""
    n_groups = len(top_groups)
    if aggregation == "equal":
        return np.ones(n_groups) / n_groups
    if aggregation == "cagr_weighted":
        raw = np.array([c for _, c, _, _, _ in top_groups])
    elif aggregation == "rtr_weighted":
        raw = np.array([r for r, _, _, _, _ in top_groups])
    elif aggregation == "stability_weighted":
        raw = np.array([s for _, _, s, _, _ in top_groups])
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
    weights = np.maximum(raw, 0.0)
    return weights / weights.sum() if weights.sum() > 0 else np.ones(n_groups) / n_groups


def slice_data(returns, tbill_returns, start, end):
    """Slice returns and cash for a date range."""
    returns_slice = returns[(returns.index >= start) & (returns.index <= end)]
    cash_slice = tbill_returns.reindex(returns_slice.index).ffill().fillna(0)
    return returns_slice, cash_slice


def generate_folds(train_start_year, first_test_year, test_years, last_year):
    """Generate walk-forward fold date ranges.

    Returns list of (train_start, train_end, test_start, test_end) date strings.
    """
    folds = []
    test_year = first_test_year
    while test_year + test_years - 1 <= last_year:
        folds.append((
            f"{train_start_year}-01-01",
            f"{test_year - 1}-12-31",
            f"{test_year}-01-01",
            f"{test_year + test_years - 1}-12-31",
        ))
        test_year += test_years
    return folds


# Transition detection

def detect_transitions(final_test, sig_test, signal_names, deltas, agreement,
                       pipe_votes, prev_sig_last, test_returns_index, fold_i):
    """Detect IN/OUT transitions in the test period.

    Returns list of transition dicts with date, direction, agreement, etc.
    """
    n_signals = len(signal_names)
    n_valid = int(np.sum(~np.isnan(pipe_votes[0])))
    transitions = []

    # Fold boundary transition
    if prev_sig_last is not None and final_test[0] != prev_sig_last:
        direction = "OUT->IN" if final_test[0] == 1 else "IN->OUT"
        off_signals = [signal_names[j] for j in range(n_signals) if sig_test[0, j] == 0]
        on_signals = [signal_names[j] for j in range(n_signals) if sig_test[0, j] == 1]
        if direction == "IN->OUT":
            changed = [f"{s}=OFF" for s in off_signals[:5]]
        else:
            changed = [f"{s}=ON" for s in on_signals[:5]]
        target = on_signals if direction == "OUT->IN" else off_signals
        if len(target) > 5:
            changed.append(f"+{len(target) - 5}")
        top_drivers_idx = np.argsort(-np.abs(deltas[0]))[:5]
        drivers = [signal_names[j] for j in top_drivers_idx if np.abs(deltas[0, j]) > 0.001]
        transitions.append({
            "date": test_returns_index[0],
            "direction": direction,
            "agreement": int(agreement[0]),
            "n_valid": n_valid,
            "votes": pipe_votes[0],
            "changed": changed if changed else [f"[fold {fold_i+1} boundary]"],
            "drivers": drivers,
            "fold_boundary": True,
        })

    # Intra-fold transitions
    for t in range(1, len(final_test)):
        if final_test[t] != final_test[t - 1]:
            direction = "OUT->IN" if final_test[t] == 1 else "IN->OUT"
            changed = []
            for j in range(n_signals):
                if sig_test[t, j] != sig_test[t - 1, j]:
                    arrow = "\u2191" if sig_test[t, j] == 1 else "\u2193"
                    changed.append(f"{signal_names[j]}{arrow}")
            top_drivers_idx = np.argsort(-np.abs(deltas[t]))[:5]
            drivers = [signal_names[j] for j in top_drivers_idx if np.abs(deltas[t, j]) > 0.001]
            transitions.append({
                "date": test_returns_index[t],
                "direction": direction,
                "agreement": int(agreement[t]),
                "n_valid": n_valid,
                "votes": pipe_votes[t],
                "changed": changed,
                "drivers": drivers,
                "fold_boundary": False,
            })

    return transitions


# Ensemble model building

def _build_ensemble_models(config, train, context,
                           adaptive_grid=None, cached_metrics=None):
    """Build ensemble models (traders + groups) for a given config."""
    trader_configs = context.trader_configs
    if adaptive_grid and ("min_signals_per_trader" in adaptive_grid or
                          "max_signals_per_trader" in adaptive_grid):
        trader_configs = make_configs(
            train.signal_matrix.shape[1], config.min_signals_per_trader, config.max_signals_per_trader
        )

    trader_rtr = float(config.trader_min_rtr)
    group_rtr = float(config.group_min_rtr)
    group_cagr = float(config.group_min_cagr)

    if cached_metrics is None:
        cached_metrics = _eval_all(train.signal_matrix, train.returns, train.cash_returns, trader_configs, context.periods_per_year)

    ensemble_models = []
    for threshold_cagr in config.cagr_thresholds:
        top_traders, _, _ = grid_search(
            train.signal_matrix, train.returns, train.cash_returns, trader_configs,
            config.top_traders, context.periods_per_year,
            trader_rtr, threshold_cagr,
            cached_metrics=cached_metrics, sort_by=config.sort_by
        )
        if not top_traders:
            ensemble_models.append(None)
            continue

        trader_signals_train = np.column_stack([
            combo_signal(train.signal_matrix, c, min_votes) for _, _, _, min_votes, c in top_traders
        ])
        group_configs = make_configs(len(top_traders), config.min_traders_per_group, config.max_traders_per_group)
        top_groups, _, _ = grid_search(
            trader_signals_train, train.returns, train.cash_returns, group_configs,
            config.top_groups, context.periods_per_year,
            group_rtr, group_cagr, sort_by=config.sort_by
        )
        if not top_groups:
            ensemble_models.append(None)
            continue

        weights = _group_weights(top_groups, config.group_aggregation)
        ensemble_models.append((top_traders, top_groups, weights))
    return ensemble_models


# Adaptive HP sweep

def _score_ensemble(ensemble_models, validation, vote_threshold, periods_per_year):
    n_valid = sum(1 for model in ensemble_models if model is not None)
    if n_valid == 0:
        return (-np.inf, -np.inf), {}
    _, _, final_val, _ = compute_ensemble_vote(validation.signal_matrix, ensemble_models, vote_threshold)
    strat_ret = compute_strategy_returns(validation.returns, final_val, validation.cash_returns)
    metrics = compute_all_metrics(strat_ret, final_val, periods_per_year, cash_returns=validation.cash_returns)
    score = (metrics["cagr"], metrics["rtr"])
    return score, metrics


def _passes_floor(metrics, floor_metrics):
    if floor_metrics is None or not metrics:
        return True
    return metrics.get("stability", -np.inf) >= floor_metrics.get("stability", -np.inf)


def _median_combo(tied_combos, hp_names):
    result = {}
    for k in hp_names:
        vals = sorted(c[k] for c in tied_combos)
        result[k] = vals[len(vals) // 2]
    return result


def _adaptive_sweep(base_config, train, val, context, adaptive_grid):
    """Sweep HPs on train->val, return (adapted PipelineConfig, sweep_data, shared_metrics)."""
    adapted = replace(base_config)
    hp_names = list(adaptive_grid.keys())
    periods_per_year = context.periods_per_year

    # Pre-compute trader metrics once - shared across all HP combos
    shared_metrics = _eval_all(train.signal_matrix, train.returns, train.cash_returns, context.trader_configs, periods_per_year)

    base_models = _build_ensemble_models(
        adapted, train, context,
        adaptive_grid, cached_metrics=shared_metrics
    )
    base_score, base_metrics = _score_ensemble(
        base_models, val, adapted.vote_threshold, periods_per_year
    )
    floor_metrics = base_metrics

    hp_values = [adaptive_grid[k] for k in hp_names]
    best_combo = {k: getattr(adapted, k) for k in hp_names}

    all_results = []
    passing_combos = []

    for combo in iterproduct(*hp_values):
        overrides = dict(zip(hp_names, combo))
        trial = replace(adapted, **overrides)
        models = _build_ensemble_models(
            trial, train, context,
            adaptive_grid, cached_metrics=shared_metrics
        )
        n_ensembles = sum(1 for model in models if model is not None)
        score, metrics = _score_ensemble(
            models, val, trial.vote_threshold, periods_per_year
        )
        passes = _passes_floor(metrics, floor_metrics)
        beats = score >= base_score
        if metrics:
            all_results.append((score, metrics, combo, overrides, passes, beats, n_ensembles))
        if passes and beats:
            passing_combos.append((score, overrides))

    if passing_combos:
        best_score = max(s for s, _ in passing_combos)
        tied = [o for s, o in passing_combos if s == best_score]
        best_combo = _median_combo(tied, hp_names) if len(tied) > 1 else tied[0]

    has_changes = any(best_combo[k] != getattr(adapted, k) for k in hp_names)
    if has_changes:
        adapted = replace(adapted, **best_combo)

    sweep_data = {
        "hp_names": hp_names,
        "all_results": all_results,
        "floor_metrics": floor_metrics,
        "base_n_ensembles": sum(1 for model in base_models if model is not None),
        "base_values": {k: getattr(base_config, k) for k in hp_names},
        "best_combo": best_combo,
        "has_changes": has_changes,
    }

    return adapted, sweep_data, shared_metrics


# OOS aggregation

def aggregate_oos(accum, tbill_returns, periods_per_year):
    """Concatenate per-fold OOS arrays and compute aggregate metrics.

    Args:
        accum: dict with keys oos_returns, oos_signal, oos_buy_hold, oos_dates,
               oos_agreement, oos_deltas, oos_raw_signals, oos_folds.
               Each value is a list of per-fold arrays.

    Returns (oos_arrays_dict, metrics_oos, metrics_buy_hold_oos).
    """
    oos_returns = np.concatenate(accum["oos_returns"])
    oos_signal = np.concatenate(accum["oos_signal"])
    oos_agreement = np.concatenate(accum["oos_agreement"])
    oos_deltas = np.vstack(accum["oos_deltas"])
    oos_raw_signals = np.vstack(accum["oos_raw_signals"])
    oos_folds = np.concatenate(accum["oos_folds"]).astype(int)
    oos_dates = pd.DatetimeIndex(np.concatenate([d.values for d in accum["oos_dates"]]))
    oos_buy_hold = np.concatenate(accum["oos_buy_hold"])

    oos_cash = tbill_returns.reindex(oos_dates).ffill().fillna(0).values
    metrics_oos = compute_all_metrics(oos_returns, oos_signal, periods_per_year, cash_returns=oos_cash)
    metrics_buy_hold_oos = compute_all_metrics(oos_buy_hold, np.ones(len(oos_buy_hold)), periods_per_year, cash_returns=oos_cash)

    arrays = {
        "oos_returns": oos_returns, "oos_signal": oos_signal, "oos_dates": oos_dates,
        "oos_agreement": oos_agreement, "oos_deltas": oos_deltas,
        "oos_raw_signals": oos_raw_signals, "oos_buy_hold": oos_buy_hold, "oos_folds": oos_folds,
    }
    return arrays, metrics_oos, metrics_buy_hold_oos


# Fold processing

_OOS_KEYS = ("oos_returns", "oos_signal", "oos_buy_hold", "oos_dates",
             "oos_agreement", "oos_deltas", "oos_raw_signals", "oos_folds")


def _failed_fold_result(returns_test, signal_test, context, fold_i, test_returns_index,
                        base_result, cash_test, ensemble_models, fold_duration):
    """Build output dict when all ensemble thresholds fail."""
    n_signals = context.n_signals
    metrics_buy_hold = compute_all_metrics(returns_test, np.ones(len(returns_test)), context.periods_per_year, cash_returns=cash_test)
    return {
        "oos_returns": returns_test, "oos_signal": np.ones(len(returns_test)),
        "oos_buy_hold": returns_test, "oos_dates": test_returns_index,
        "oos_agreement": np.full(len(returns_test), 0),
        "oos_deltas": np.zeros((len(returns_test), n_signals)),
        "oos_raw_signals": signal_test,
        "oos_folds": np.full(len(returns_test), fold_i + 1),
        "fold_result": {
            **base_result,
            "test_cagr": metrics_buy_hold["cagr"], "test_rtr": metrics_buy_hold["rtr"],
            "test_sharpe": metrics_buy_hold["sharpe"], "test_max_drawdown": metrics_buy_hold["max_drawdown"],
            "buy_hold_cagr": metrics_buy_hold["cagr"], "buy_hold_rtr": metrics_buy_hold["rtr"],
            "buy_hold_sharpe": metrics_buy_hold["sharpe"], "buy_hold_max_drawdown": metrics_buy_hold["max_drawdown"],
            "transitions": [],
            "structural_importance": np.zeros(n_signals), "percent_delta": np.zeros(n_signals),
            "metrics_train": None, "metrics_buy_hold_train": None,
            "train_exposure": None, "test_exposure": 1.0,
            "all_failed": True, "duration": fold_duration,
        },
        "structural_importance": np.zeros(n_signals),
        "ensemble_models": ensemble_models,
    }


def _process_fold(context, fold_i, fold_dates, prev_sig_last):
    """Process a single walk-forward fold.

    Returns dict with OOS arrays, fold_result, struct_imp, ensemble_models.
    """
    train_start, train_end, test_start, test_end = fold_dates
    time_fold_start = time.time()
    config = context.config
    n_signals = context.n_signals
    periods_per_year = context.periods_per_year

    train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
    test_returns, cash_test = slice_data(context.returns, context.tbill_returns, test_start, test_end)

    signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
    signal_test = build_sig_matrix(context.signals, context.signal_names, test_returns.index)

    returns_train = train_returns.values.astype(np.float64)
    cash_train = cash_train.values.astype(np.float64)
    returns_test = test_returns.values.astype(np.float64)
    cash_test = cash_test.values.astype(np.float64)
    train = SplitData(signal_train, returns_train, cash_train)

    # --- Adaptive HP sweep ---
    fold_config = config
    adaptive_grid = config.adaptive_grid
    sweep_data = None
    periods_per_year_int = int(periods_per_year)
    if adaptive_grid and len(returns_train) > config.test_years * periods_per_year_int + periods_per_year_int:
        if config.adaptive_val_years == "full":
            inner_train = train
            inner_val = train
        else:
            val_size = int(config.adaptive_val_years * periods_per_year)
            inner_train = SplitData(signal_train[:-val_size], returns_train[:-val_size], cash_train[:-val_size])
            inner_val = SplitData(signal_train[-val_size:], returns_train[-val_size:], cash_train[-val_size:])

        fold_config, sweep_data, sweep_metrics = _adaptive_sweep(
            config, inner_train, inner_val, context, adaptive_grid
        )

    # --- Build ensemble ---
    if sweep_data is not None and config.adaptive_val_years == "full":
        trader_metrics = sweep_metrics
    else:
        trader_metrics = _eval_all(train.signal_matrix, train.returns, train.cash_returns, context.trader_configs, periods_per_year)
    ensemble_models = _build_ensemble_models(
        fold_config, train, context,
        cached_metrics=trader_metrics
    )

    n_valid = sum(1 for model in ensemble_models if model is not None)
    period = f"{test_start[:4]}-{test_end[:4]}"
    base_result = {
        "fold": fold_i + 1, "period": period,
        "train_start": train_start[:4], "train_end": train_end[:4],
        "adapted_hps": {k: getattr(fold_config, k) for k in adaptive_grid} if adaptive_grid else {},
        "sweep_data": sweep_data,
    }

    if n_valid == 0:
        return _failed_fold_result(
            returns_test, signal_test, context, fold_i, test_returns.index,
            base_result, cash_test, ensemble_models, time.time() - time_fold_start
        )

    _, pipe_votes, final_test, agreement = compute_ensemble_vote(
        signal_test, ensemble_models, fold_config.vote_threshold
    )
    structural_importance = structural_importance_ensemble(context.signal_names, ensemble_models)

    deltas = signal_importance_ensemble(
        signal_test, context.signal_names, ensemble_models,
        fold_config.vote_threshold, final_test
    )

    avg_delta = np.mean(np.abs(deltas), axis=0)
    total_delta = avg_delta.sum()
    percent_delta = avg_delta / total_delta * 100 if total_delta > 0 else np.zeros(n_signals)

    # --- Evaluate ---
    _, _, final_train, _ = compute_ensemble_vote(signal_train, ensemble_models, fold_config.vote_threshold)
    train_strategy_returns = compute_strategy_returns(returns_train, final_train, cash_train)
    metrics_train = compute_all_metrics(train_strategy_returns, final_train, periods_per_year, cash_returns=cash_train)
    metrics_buy_hold_train = compute_all_metrics(returns_train, np.ones(len(returns_train)), periods_per_year, cash_returns=cash_train)
    test_strategy_returns = compute_strategy_returns(returns_test, final_test, cash_test)
    metrics_test = compute_all_metrics(test_strategy_returns, final_test, periods_per_year, cash_returns=cash_test)
    metrics_buy_hold = compute_all_metrics(returns_test, np.ones(len(returns_test)), periods_per_year, cash_returns=cash_test)

    # --- Transitions ---
    transitions = detect_transitions(
        final_test, signal_test, context.signal_names, deltas, agreement,
        pipe_votes, prev_sig_last, test_returns.index, fold_i
    )

    fold_duration = time.time() - time_fold_start
    return {
        "oos_returns": test_strategy_returns, "oos_signal": final_test,
        "oos_buy_hold": returns_test, "oos_dates": test_returns.index,
        "oos_agreement": agreement, "oos_deltas": deltas,
        "oos_raw_signals": signal_test,
        "oos_folds": np.full(len(returns_test), fold_i + 1),
        "fold_result": {
            **base_result,
            "test_rtr": metrics_test["rtr"], "test_sharpe": metrics_test["sharpe"],
            "test_cagr": metrics_test["cagr"], "test_max_drawdown": metrics_test["max_drawdown"],
            "buy_hold_cagr": metrics_buy_hold["cagr"], "buy_hold_rtr": metrics_buy_hold["rtr"],
            "buy_hold_sharpe": metrics_buy_hold["sharpe"], "buy_hold_max_drawdown": metrics_buy_hold["max_drawdown"],
            "transitions": transitions,
            "structural_importance": structural_importance, "percent_delta": percent_delta,
            "metrics_train": metrics_train, "metrics_buy_hold_train": metrics_buy_hold_train,
            "train_exposure": float(final_train.mean()),
            "test_exposure": float(final_test.mean()),
            "all_failed": False, "duration": fold_duration,
        },
        "structural_importance": structural_importance,
        "ensemble_models": ensemble_models,
    }


# Main pipeline

def run_pipeline(config, signals, signal_names, returns, tbill_returns, periods_per_year, last_year):
    """Run the full walk-forward consensus pipeline.

    Returns dict with all OOS results, metrics, fold data, and explainability.
    No printing - caller handles display.
    """
    n_signals = len(signal_names)

    folds = generate_folds(config.train_start_year, config.first_test_year, config.test_years,
                           last_year)
    if not folds:
        return None

    trader_configs = make_configs(n_signals, config.min_signals_per_trader, config.max_signals_per_trader)
    context = _Context(config, signals, signal_names, n_signals, returns, tbill_returns, trader_configs, periods_per_year)

    accum = {k: [] for k in _OOS_KEYS}
    fold_results = []
    fold_structural_importance = []
    trader_counts = Counter()
    prev_ensemble = None
    prev_sig_last = None
    time_total_start = time.time()

    for fold_i, fold_dates in enumerate(folds):
        out = _process_fold(context, fold_i, fold_dates, prev_sig_last)

        for k in _OOS_KEYS:
            accum[k].append(out[k])
        fold_results.append(out["fold_result"])
        fold_structural_importance.append(out["structural_importance"])

        # Track trader usage
        for model in out["ensemble_models"]:
            if model is None:
                continue
            top_traders, _, _ = model
            for _, _, _, min_votes, combo in top_traders:
                names = tuple(signal_names[i] for i in combo)
                trader_counts[(min_votes, names)] += 1

        prev_ensemble = out["ensemble_models"]
        prev_sig_last = out["oos_signal"][-1] if len(out["oos_signal"]) > 0 else None

    # --- Aggregate OOS ---
    total_duration = time.time() - time_total_start

    arrays, metrics_oos, metrics_buy_hold_oos = aggregate_oos(accum, tbill_returns, periods_per_year)

    oos_cash = tbill_returns.reindex(arrays["oos_dates"]).ffill().fillna(0).values
    pnl_gain, pnl_cost, pnl_net = signal_pnl_attribution(
        arrays["oos_deltas"], arrays["oos_buy_hold"], oos_cash
    )

    last_structural_importance = structural_importance_ensemble(signal_names, prev_ensemble)

    return {
        "cagr": metrics_oos["cagr"],
        "rtr": metrics_oos["rtr"],
        "sharpe": metrics_oos["sharpe"],
        "max_drawdown": metrics_oos["max_drawdown"],
        "exposure": float(arrays["oos_signal"].mean()),
        **arrays,
        "metrics_oos": metrics_oos,
        "metrics_buy_hold_oos": metrics_buy_hold_oos,
        "fold_results": fold_results,
        "trader_counts": trader_counts,
        "total_duration": total_duration,
        "folds": folds,
        "cagr_thresholds": config.cagr_thresholds,
        "last_structural_importance": last_structural_importance,
        "fold_structural_importance": fold_structural_importance,
        "pnl_gain": pnl_gain,
        "pnl_cost": pnl_cost,
        "pnl_net": pnl_net,
    }
