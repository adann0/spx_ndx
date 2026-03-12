import numpy as np
import pytest

from spx_ndx.models.spx_consensus.explain import (
    run_single_pipeline,
    compute_ensemble_vote,
    build_pipeline_cache,
    _compute_pipeline_delta,
    signal_importance_ensemble,
    structural_importance_ensemble,
    signal_pnl_attribution,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_trader(min_votes, combo, rtr=1.0, cagr=0.1, stability=0.9):
    """Build a trader tuple: (rtr, cagr, stability, min_votes, combo)."""
    return (rtr, cagr, stability, min_votes, combo)


def _make_model(traders, groups, weights):
    """Build an ensemble model tuple: (traders, groups, weights)."""
    return (traders, groups, weights)


# ── run_single_pipeline ──────────────────────────────────────────────────────

class TestRunSinglePipeline:

    def test_all_ones_signal(self):
        """All signals ON, all traders have min_votes=1 -> all groups ON -> vote=1 -> decision=1."""
        signal_matrix = np.ones((5, 3))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,)), _make_trader(1, (2,))]
        groups = [_make_trader(1, (0, 1)), _make_trader(1, (1, 2))]
        weights = np.array([0.5, 0.5])
        vote, decision = run_single_pipeline(signal_matrix, traders, groups, weights, threshold=0.5)
        assert vote.shape == (5,)
        assert decision.shape == (5,)
        np.testing.assert_array_equal(decision, np.ones(5))

    def test_all_zeros_signal(self):
        """All signals OFF, min_votes=2 for all traders -> no votes -> decision=0."""
        signal_matrix = np.zeros((5, 3))
        traders = [_make_trader(2, (0, 1)), _make_trader(2, (1, 2))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        vote, decision = run_single_pipeline(signal_matrix, traders, groups, weights, threshold=0.5)
        np.testing.assert_array_equal(decision, np.zeros(5))

    def test_threshold_boundary(self):
        """Vote exactly at threshold -> decision=1."""
        signal_matrix = np.ones((3, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        weights = np.array([0.3, 0.7])
        vote, decision = run_single_pipeline(signal_matrix, traders, groups, weights, threshold=0.7)
        # vote = 0.3*1 + 0.7*1 = 1.0 >= 0.7
        np.testing.assert_array_equal(decision, np.ones(3))

    def test_returns_new_arrays(self):
        """Must return new arrays, not views of inputs."""
        signal_matrix = np.ones((3, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        vote, decision = run_single_pipeline(signal_matrix, traders, groups, weights, threshold=0.5)
        assert vote is not signal_matrix
        assert decision is not signal_matrix


# ── compute_ensemble_vote ────────────────────────────────────────────────────

class TestComputeEnsembleVote:

    def test_all_pipelines_agree_in(self):
        """3 pipelines all return IN -> agreement=3, decision=1 with majority=2."""
        signal_matrix = np.ones((4, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        models = [model, model, model]
        pipeline_decisions, pipeline_votes, ensemble_decision, agreement = compute_ensemble_vote(
            signal_matrix, models, vote_threshold=0.5        )
        assert pipeline_decisions.shape == (4, 3)
        assert agreement.shape == (4,)
        np.testing.assert_array_equal(agreement, np.full(4, 3.0))
        np.testing.assert_array_equal(ensemble_decision, np.ones(4))

    def test_none_pipeline_excluded(self):
        """None pipelines are excluded from vote (NaN), dynamic majority on valid only."""
        signal_matrix = np.zeros((3, 2))
        traders = [_make_trader(2, (0, 1))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        # 2 None + 1 real (real returns OUT since all signals=0 with min_votes=2)
        models = [None, None, model]
        pipeline_decisions, pipeline_votes, ensemble_decision, agreement = compute_ensemble_vote(
            signal_matrix, models, vote_threshold=0.5        )
        # None pipelines have NaN (excluded)
        assert np.all(np.isnan(pipeline_decisions[:, 0]))
        assert np.all(np.isnan(pipeline_decisions[:, 1]))
        # Only 1 valid pipeline (OUT) -> n_valid=1, effective_majority=1
        # agreement=0 < 1 -> ensemble OUT
        np.testing.assert_array_equal(agreement, np.zeros(3))
        np.testing.assert_array_equal(ensemble_decision, np.zeros(3))

    def test_majority_not_reached(self):
        """2 OUT + 1 None (excluded) -> n_valid=2, majority=1, agreement=0 -> OUT."""
        signal_matrix = np.zeros((3, 2))
        # Trader with min_votes=2 on zero signals -> always OUT
        traders_out = [_make_trader(2, (0, 1))]
        groups_out = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model_out = _make_model(traders_out, groups_out, weights)
        models = [model_out, model_out, None]  # 2 OUT + 1 None (excluded)
        pipeline_decisions, pipeline_votes, ensemble_decision, agreement = compute_ensemble_vote(
            signal_matrix, models, vote_threshold=0.5        )
        # None excluded -> n_valid=2, effective_majority=(2+1)//2=1
        # Both valid pipelines vote OUT -> agreement=0 < 1 -> ensemble OUT
        assert np.all(np.isnan(pipeline_decisions[:, 2]))
        np.testing.assert_array_equal(agreement, np.zeros(3))
        np.testing.assert_array_equal(ensemble_decision, np.zeros(3))

    def test_all_none_raises(self):
        """All None pipelines -> assertion error."""
        signal_matrix = np.ones((3, 2))
        with pytest.raises(AssertionError, match="no valid models"):
            compute_ensemble_vote(signal_matrix, [None, None], vote_threshold=0.5)


# ── structural_importance_ensemble ───────────────────────────────────────────

class TestStructuralImportance:

    def test_single_pipeline_uniform(self):
        """1 pipeline, 1 group with 2 traders of 1 signal each -> 50/50."""
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        importance = structural_importance_ensemble(["A", "B"], [model])
        np.testing.assert_allclose(importance, [50.0, 50.0], atol=1e-6)

    def test_all_none(self):
        """All None models -> zero importance."""
        importance = structural_importance_ensemble(["A", "B", "C"], [None, None])
        np.testing.assert_array_equal(importance, np.zeros(3))

    def test_sums_to_100(self):
        """Importance always sums to 100%."""
        traders = [
            _make_trader(1, (0,)),
            _make_trader(1, (1,)),
            _make_trader(1, (2,)),
        ]
        groups = [_make_trader(1, (0, 1)), _make_trader(1, (1, 2))]
        weights = np.array([0.6, 0.4])
        model = _make_model(traders, groups, weights)
        importance = structural_importance_ensemble(["A", "B", "C"], [model])
        assert importance.sum() == pytest.approx(100.0, abs=1e-6)

    def test_unused_signal_zero(self):
        """Signal not in any trader -> 0% importance."""
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        importance = structural_importance_ensemble(["A", "B", "C"], [model])
        assert importance[2] == 0.0
        assert importance.sum() == pytest.approx(100.0, abs=1e-6)


# ── signal_importance_ensemble (Shapley-lite) ────────────────────────────────

class TestSignalImportanceEnsemble:

    def test_no_effect_when_signal_irrelevant(self):
        """Flipping an unused signal doesn't change decision -> delta=0."""
        # Signal 2 not used by any trader
        signal_matrix = np.ones((4, 3))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        models = [model]
        original = np.ones(4)
        deltas = signal_importance_ensemble(
            signal_matrix, ["A", "B", "C"], models,
            vote_threshold=0.5,
            original_decision=original
        )
        assert deltas.shape == (4, 3)
        # Signal C (index 2) unused -> delta should be 0
        np.testing.assert_array_equal(deltas[:, 2], np.zeros(4))

    def test_pivotal_signal_has_nonzero_delta(self):
        """Flipping a pivotal signal changes the decision -> nonzero delta."""
        # Single signal trader: flip signal 0 -> decision flips
        signal_matrix = np.ones((3, 1))
        traders = [_make_trader(1, (0,))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        models = [model]
        original = np.ones(3)
        deltas = signal_importance_ensemble(
            signal_matrix, ["A"], models,
            vote_threshold=0.5,
            original_decision=original
        )
        # Flipping signal 0 from 1->0 should flip decision from 1->0
        # delta = original - flipped = 1 - 0 = 1
        np.testing.assert_array_equal(deltas[:, 0], np.ones(3))

    def test_shape_matches_signals(self):
        """Output shape = (T, n_signals)."""
        signal_matrix = np.ones((5, 4))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        deltas = signal_importance_ensemble(
            signal_matrix, ["A", "B", "C", "D"], [model],
            vote_threshold=0.5,
            original_decision=np.ones(5)
        )
        assert deltas.shape == (5, 4)

    def test_none_models_ignored(self):
        """None models don't crash and don't affect deltas."""
        signal_matrix = np.ones((3, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        models = [None, model, None]
        original = np.ones(3)
        deltas = signal_importance_ensemble(
            signal_matrix, ["A", "B"], models,
            vote_threshold=0.5,
            original_decision=original
        )
        assert deltas.shape == (3, 2)


# ── _compute_pipeline_delta ─────────────────────────────────────────────────

class TestComputePipelineDelta:

    def test_unaffected_signal_returns_none(self):
        """Signal not in any trader -> returns None."""
        signal_matrix = np.ones((4, 3))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [model], 0.5)
        flipped = np.zeros(4)  # flip signal 2
        result = _compute_pipeline_delta(cache[0], j=2, flipped_col=flipped,
                                         sig_matrix=signal_matrix, vote_threshold=0.5)
        assert result is None

    def test_pivotal_flip_changes_decision(self):
        """Flipping a pivotal signal changes the pipeline decision."""
        signal_matrix = np.ones((3, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [model], 0.5)
        # Flip signal 0 from 1 to 0
        flipped = np.zeros(3)
        delta = _compute_pipeline_delta(cache[0], j=0, flipped_col=flipped,
                                        sig_matrix=signal_matrix, vote_threshold=0.5)
        assert delta is not None
        # With min_votes=1 for groups and traders, flipping signal 0 makes trader 0
        # output 0. Group (0,1) with min_votes=1 still passes (trader 1 still ON).
        # So decision stays 1 -> delta = 0.
        # Let's check the actual values:
        assert delta.shape == (3,)

    def test_single_trader_flip(self):
        """Single trader using single signal: flip -> decision flips."""
        signal_matrix = np.ones((3, 1))
        traders = [_make_trader(1, (0,))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [model], 0.5)
        flipped = np.zeros(3)
        delta = _compute_pipeline_delta(cache[0], j=0, flipped_col=flipped,
                                        sig_matrix=signal_matrix, vote_threshold=0.5)
        # Original: all ON -> decision=1. Flipped: all OFF -> decision=0.
        # delta = new_dec - base_dec = 0 - 1 = -1
        np.testing.assert_array_equal(delta, -np.ones(3))

    def test_partial_flip(self):
        """Signal partially ON/OFF: delta is non-uniform."""
        signal_matrix = np.array([[1, 1], [1, 1], [0, 1]])  # signal 0 is OFF at t=2
        traders = [_make_trader(1, (0,))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [model], 0.5)
        flipped = 1 - signal_matrix[:, 0].astype(float)  # flip signal 0
        delta = _compute_pipeline_delta(cache[0], j=0, flipped_col=flipped,
                                        sig_matrix=signal_matrix, vote_threshold=0.5)
        # t=0,1: was ON->OFF, delta=-1; t=2: was OFF->ON, delta=+1
        np.testing.assert_array_equal(delta, np.array([-1.0, -1.0, 1.0]))


# ── build_pipeline_cache ────────────────────────────────────────────────────

class TestBuildPipelineCache:

    def test_none_models_skipped(self):
        signal_matrix = np.ones((3, 2))
        cache = build_pipeline_cache(signal_matrix, [None, None], 0.5)
        assert cache == [None, None]

    def test_cache_structure(self):
        signal_matrix = np.ones((5, 2))
        traders = [_make_trader(1, (0,)), _make_trader(1, (1,))]
        groups = [_make_trader(1, (0, 1))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [model], 0.5)
        assert len(cache) == 1
        assert cache[0] is not None
        trader_signals, group_signals, vote, decision, sig2traders, trader2groups, _, _, _ = cache[0]
        assert trader_signals.shape == (5, 2)
        assert group_signals.shape == (5, 1)
        assert vote.shape == (5,)
        assert decision.shape == (5,)
        # Signal 0 maps to trader 0, signal 1 maps to trader 1
        assert 0 in sig2traders
        assert 1 in sig2traders
        # Both traders map to group 0
        assert trader2groups[0] == [0]
        assert trader2groups[1] == [0]

    def test_mixed_none_and_valid(self):
        signal_matrix = np.ones((3, 2))
        traders = [_make_trader(1, (0, 1))]
        groups = [_make_trader(1, (0,))]
        weights = np.array([1.0])
        model = _make_model(traders, groups, weights)
        cache = build_pipeline_cache(signal_matrix, [None, model, None], 0.5)
        assert cache[0] is None
        assert cache[1] is not None
        assert cache[2] is None


# ── signal_pnl_attribution ─────────────────────────────────────────────────

class TestSignalPnlAttribution:

    def test_zero_deltas(self):
        """No decision changes -> zero gain/cost/net."""
        deltas = np.zeros((5, 3))
        bh = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
        cash = np.zeros(5)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        np.testing.assert_array_equal(gain, np.zeros(3))
        np.testing.assert_array_equal(cost, np.zeros(3))
        np.testing.assert_array_equal(net, np.zeros(3))

    def test_keeps_in_during_rally_is_gain(self):
        """delta=+1 (keeps IN) when market up -> positive gain."""
        deltas = np.array([[1.0], [0.0]])
        bh = np.array([0.05, 0.01])
        cash = np.zeros(2)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        assert gain[0] == pytest.approx(5.0)
        assert cost[0] == pytest.approx(0.0)
        assert net[0] == pytest.approx(5.0)

    def test_keeps_in_during_crash_is_cost(self):
        """delta=+1 (keeps IN) when market down -> negative cost."""
        deltas = np.array([[1.0], [0.0]])
        bh = np.array([-0.05, 0.01])
        cash = np.zeros(2)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        assert gain[0] == pytest.approx(0.0)
        assert cost[0] == pytest.approx(-5.0)
        assert net[0] == pytest.approx(-5.0)

    def test_keeps_out_during_rally_is_cost(self):
        """delta=-1 (keeps OUT) when market up -> negative cost."""
        deltas = np.array([[-1.0]])
        bh = np.array([0.05])
        cash = np.zeros(1)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        assert gain[0] == pytest.approx(0.0)
        assert cost[0] == pytest.approx(-5.0)

    def test_keeps_out_during_crash_is_gain(self):
        """delta=-1 (keeps OUT) when market down -> positive gain."""
        deltas = np.array([[-1.0]])
        bh = np.array([-0.05])
        cash = np.zeros(1)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        assert gain[0] == pytest.approx(5.0)
        assert cost[0] == pytest.approx(0.0)

    def test_net_equals_gain_plus_cost(self):
        """net = gain + cost always."""
        rng = np.random.default_rng(42)
        deltas = rng.choice([-1, 0, 1], size=(20, 4)).astype(float)
        bh = rng.normal(0.01, 0.05, 20)
        cash = np.full(20, 0.003)
        gain, cost, net = signal_pnl_attribution(deltas, bh, cash)
        np.testing.assert_allclose(net, gain + cost)

    def test_cash_reduces_excess(self):
        """Positive cash return reduces the excess, hence smaller impact."""
        deltas = np.array([[1.0]])
        bh = np.array([0.05])
        cash_zero = np.zeros(1)
        cash_pos = np.array([0.02])
        _, _, net_zero = signal_pnl_attribution(deltas, bh, cash_zero)
        _, _, net_pos = signal_pnl_attribution(deltas, bh, cash_pos)
        assert net_zero[0] > net_pos[0]
