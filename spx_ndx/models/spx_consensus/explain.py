"""Explainability functions for the consensus ensemble pipeline.

Pure functions - no I/O, no global state, no side effects.
"""

import numpy as np

from spx_ndx.models.spx_consensus.combo import combo_signal, majority_vote


def run_single_pipeline(sig_matrix, top_traders, top_groups, weights, threshold):
    """Run one pipeline: signals -> traders -> groups -> weighted vote -> binary."""
    trader_signals = np.column_stack([combo_signal(sig_matrix, c, mv) for _, _, _, mv, c in top_traders])
    group_signals = np.column_stack([combo_signal(trader_signals, c, mv) for _, _, _, mv, c in top_groups])
    vote = group_signals @ weights
    decision = (vote >= threshold).astype(np.float64)
    return vote, decision


def compute_ensemble_vote(sig_matrix, ensemble_models, vote_threshold):
    """Run all pipelines, return per-pipeline decisions and ensemble decision.

    None pipelines (failed CAGR thresholds) are excluded from the vote.
    Majority = (n_valid + 1) // 2.

    Returns:
        pipeline_decisions: (n_timesteps, n_pipelines) array (NaN for None pipelines)
        pipeline_votes: (n_timesteps, n_pipelines) array (NaN for None pipelines)
        ensemble_decision: (n_timesteps,) final binary decision
        agreement: (n_timesteps,) number of valid pipelines voting IN
    """
    n_timesteps = sig_matrix.shape[0]
    n_pipelines = len(ensemble_models)
    pipeline_decisions = np.full((n_timesteps, n_pipelines), np.nan)
    pipeline_votes = np.full((n_timesteps, n_pipelines), np.nan)

    n_valid = 0
    for k, model in enumerate(ensemble_models):
        if model is None:
            continue
        n_valid += 1
        top_traders, top_groups, weights = model
        vote, decision = run_single_pipeline(sig_matrix, top_traders, top_groups, weights, vote_threshold)
        pipeline_decisions[:, k] = decision
        pipeline_votes[:, k] = vote

    assert n_valid > 0, "All ensemble pipelines failed - no valid models"

    effective_majority = (n_valid + 1) // 2
    agreement = np.nansum(pipeline_decisions, axis=1)
    ensemble_decision = (agreement >= effective_majority).astype(np.float64)
    return pipeline_decisions, pipeline_votes, ensemble_decision, agreement


def build_pipeline_cache(sig_matrix, ensemble_models, vote_threshold):
    """Pre-compute base state for each pipeline in the ensemble.

    Returns list (one per pipeline, None for failed pipelines) of tuples:
        (trader_signals, group_signals, vote, decision, sig2traders, trader2groups, top_traders, top_groups, weights)
    """
    cache = []
    for model in ensemble_models:
        if model is None:
            cache.append(None)
            continue

        top_traders, top_groups, weights = model
        trader_signals = np.column_stack([combo_signal(sig_matrix, c, mv) for _, _, _, mv, c in top_traders])
        group_signals = np.column_stack([combo_signal(trader_signals, c, mv) for _, _, _, mv, c in top_groups])
        vote = group_signals @ weights
        decision = (vote >= vote_threshold).astype(np.float64)

        sig2traders = {}
        for trader_index, (_, _, _, _, combo) in enumerate(top_traders):
            for signal_index in combo:
                if signal_index not in sig2traders:
                    sig2traders[signal_index] = []
                sig2traders[signal_index].append(trader_index)

        trader2groups = {}
        for group_index, (_, _, _, _, combo) in enumerate(top_groups):
            for trader_index in combo:
                if trader_index not in trader2groups:
                    trader2groups[trader_index] = []
                trader2groups[trader_index].append(group_index)

        cache.append((trader_signals, group_signals, vote, decision, sig2traders, trader2groups,
                       top_traders, top_groups, weights))
    return cache


def _compute_pipeline_delta(cache_entry, j, flipped_col, sig_matrix, vote_threshold):
    """Compute per-pipeline decision delta when flipping signal j.

    Returns the change in decision (new_decision - base_decision) as a (n_timesteps,) array,
    or None if signal j does not affect this pipeline.
    """
    trader_signals, group_signals, vote, decision, sig2traders, trader2groups, \
        top_traders, top_groups, weights = cache_entry

    affected_traders = sig2traders.get(j, [])
    if not affected_traders:
        return None

    n_timesteps = sig_matrix.shape[0]

    # Recompute affected trader signals with the flipped column
    new_trader_signals = {}
    for trader_index in affected_traders:
        _, _, _, trader_min_votes, trader_combo = top_traders[trader_index]
        selected = np.empty((n_timesteps, len(trader_combo)))
        for combo_index, signal_index in enumerate(trader_combo):
            selected[:, combo_index] = flipped_col if signal_index == j else sig_matrix[:, signal_index]
        new_trader_signals[trader_index] = majority_vote(selected, trader_min_votes)

    # Find groups that depend on any affected trader
    affected_groups = set()
    for trader_index in affected_traders:
        for group_index in trader2groups.get(trader_index, []):
            affected_groups.add(group_index)

    # Recompute affected group votes
    vote_delta = np.zeros(n_timesteps)
    for group_index in affected_groups:
        _, _, _, group_min_votes, group_combo = top_groups[group_index]
        selected = np.empty((n_timesteps, len(group_combo)))
        for combo_index, trader_index in enumerate(group_combo):
            selected[:, combo_index] = new_trader_signals[trader_index] if trader_index in new_trader_signals else trader_signals[:, trader_index]
        new_group_column = majority_vote(selected, group_min_votes)
        vote_delta += (new_group_column - group_signals[:, group_index]) * weights[group_index]

    new_vote = vote + vote_delta
    new_decision = (new_vote >= vote_threshold).astype(np.float64)
    return new_decision - decision


def signal_importance_ensemble(sig_matrix, signal_names, ensemble_models,
                               vote_threshold, original_decision):
    """Shapley-lite: flip each signal, re-run all pipelines, measure decision change."""
    n_signals = len(signal_names)
    n_timesteps = sig_matrix.shape[0]
    n_pipelines = len(ensemble_models)

    cache = build_pipeline_cache(sig_matrix, ensemble_models, vote_threshold)

    n_valid = sum(1 for c in cache if c is not None)
    assert n_valid > 0, "All ensemble pipelines failed - no valid models"
    effective_majority = (n_valid + 1) // 2

    base_decisions = [c[3] for c in cache if c is not None]
    base_agreement = np.sum(base_decisions, axis=0)

    # For each signal j, compute flipped ensemble decision
    deltas = np.zeros((n_timesteps, n_signals))

    for j in range(n_signals):
        flipped_col = 1.0 - sig_matrix[:, j]
        new_agreement = base_agreement.copy()

        for k in range(n_pipelines):
            if cache[k] is None:
                continue

            delta = _compute_pipeline_delta(
                cache[k], j, flipped_col, sig_matrix, vote_threshold
            )
            if delta is not None:
                new_agreement += delta

        flipped_ensemble = (new_agreement >= effective_majority).astype(np.float64)
        deltas[:, j] = original_decision - flipped_ensemble

    return deltas


def structural_importance_ensemble(signal_names, ensemble_models):
    """Average structural importance across all ensemble pipelines.

    Sums to 100% (normalized). Returns array of shape (n_signals,).
    """
    n_signals = len(signal_names)
    total_importance = np.zeros(n_signals)
    n_valid = 0

    for model in ensemble_models:
        if model is None:
            continue
        top_traders, top_groups, weights = model
        importance = np.zeros(n_signals)
        for group_index, (_, _, _, group_min_votes, group_combo) in enumerate(top_groups):
            group_weight = weights[group_index]
            for trader_index in group_combo:
                _, _, _, trader_min_votes, trader_combo = top_traders[trader_index]
                for signal_index in trader_combo:
                    importance[signal_index] += group_weight / (len(group_combo) * len(trader_combo))
        total = importance.sum()
        if total > 0:
            importance = importance / total * 100
        total_importance += importance
        n_valid += 1

    if n_valid > 0:
        total_importance /= n_valid
    return total_importance


def signal_pnl_attribution(deltas, buy_hold_returns, cash_returns):
    """PnL attribution per signal: gain, cost, net in percentage points.

    For each signal j at each timestep t:
        pnl = delta[t,j] * (buy_hold_return[t] - cash_return[t])

    delta=+1 means signal keeps us IN; delta=-1 means signal keeps us OUT.
    Positive pnl = good decision, negative pnl = bad decision.

    Returns (gain, cost, net) arrays of shape (n_signals,) in pp.
    """
    excess = buy_hold_returns - cash_returns
    pnl = deltas * excess[:, None]
    gain = np.sum(np.maximum(pnl, 0), axis=0) * 100
    cost = np.sum(np.minimum(pnl, 0), axis=0) * 100
    net = gain + cost
    return gain, cost, net
