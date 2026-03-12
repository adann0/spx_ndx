"""Integration tests for pipeline.py using a frozen dataset snapshot."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spx_ndx.models.spx_consensus.config import PipelineConfig
from spx_ndx.models.spx_consensus.signals import compute_signals, build_sig_matrix
from spx_ndx.models.spx_consensus.grid import make_configs, warmup, _eval_all
from spx_ndx.models.spx_consensus.pipeline import (
    _build_ensemble_models, _adaptive_sweep, _process_fold,
    run_pipeline, generate_folds, SplitData, _Context,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "dataset_monthly.parquet"


# Fixtures - shared across all tests

@pytest.fixture(scope="module")
def dataset():
    ds = pd.read_parquet(_FIXTURE)
    ds.index = pd.to_datetime(ds.index)
    return ds


@pytest.fixture(scope="module")
def small_cfg():
    """Minimal config: few signals, 2 thresholds, small grid."""
    return PipelineConfig(
        train_start_year=1993,
        first_test_year=2005,
        test_years=2,
        trader_min_rtr=0.3,
        vote_threshold=0.6,
        min_signals_per_trader=2,
        max_signals_per_trader=3,
        top_traders=20,
        min_traders_per_group=2,
        max_traders_per_group=2,
        group_min_rtr=0.01,
        group_min_cagr=0.01,
        top_groups=10,
        group_aggregation="equal",
        cagr_thresholds=(0.07, 0.09),
        indicators={
            "SMA Xm": [10, 12],
            "VIX<X": [30],
            "RSI<X": [70],
            "CPI<X": [5],
            "EMA200>X": [0],
        },
        adaptive_val_years=0.5,
        adaptive_grid={},
    )


@pytest.fixture(scope="module")
def pipeline_data(dataset, small_cfg):
    """Compute signals, returns, and pipeline context."""
    ds = dataset
    periods_per_year = 12.0
    closes = ds["spx_close"]
    returns = closes.pct_change().dropna()
    tbill_returns = (1 + ds["tbill_rate"] / 100) ** (1 / periods_per_year) - 1

    sigs = compute_signals(ds, closes, small_cfg.indicators)
    signals = {k: v for k, v in sigs.items() if not v.isna().all()}
    signal_names = list(signals.keys())
    n_signals = len(signal_names)

    warmup()
    trader_configs = make_configs(n_signals, small_cfg.min_signals_per_trader,
                                  small_cfg.max_signals_per_trader)
    context = _Context(small_cfg, signals, signal_names, n_signals, returns, tbill_returns, trader_configs, periods_per_year)

    return {
        "ds": ds, "returns": returns, "tbill_returns": tbill_returns,
        "signals": signals, "signal_names": signal_names,
        "n_signals": n_signals, "context": context, "periods_per_year": periods_per_year,
    }


# Tests

class TestBuildEnsembleModels:
    def test_returns_list_of_correct_length(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        train = SplitData(signal_train, train_returns.values.astype(np.float64),
                          cash_train.values.astype(np.float64))

        models = _build_ensemble_models(config, train, context)
        assert isinstance(models, list)
        assert len(models) == len(config.cagr_thresholds)

    def test_models_have_traders_groups_weights(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        train = SplitData(signal_train, train_returns.values.astype(np.float64),
                          cash_train.values.astype(np.float64))

        models = _build_ensemble_models(config, train, context)
        for model in models:
            if model is None:
                continue
            top_traders, top_groups, weights = model
            assert len(top_traders) > 0
            assert len(top_groups) > 0
            assert len(weights) == len(top_groups)
            assert pytest.approx(weights.sum()) == 1.0

    def test_cached_metrics_same_result(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        train = SplitData(signal_train, train_returns.values.astype(np.float64),
                          cash_train.values.astype(np.float64))

        cached = _eval_all(train.signal_matrix, train.returns, train.cash_returns, context.trader_configs, context.periods_per_year)
        models_fresh = _build_ensemble_models(config, train, context)
        models_cached = _build_ensemble_models(config, train, context, cached_metrics=cached)

        assert len(models_fresh) == len(models_cached)
        for mf, mc in zip(models_fresh, models_cached):
            if mf is None:
                assert mc is None
            else:
                # Same traders selected
                assert len(mf[0]) == len(mc[0])


class TestAdaptiveSweep:
    def test_returns_config_and_sweep_data(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        periods_per_year = context.periods_per_year
        adaptive_grid = {"vote_threshold": [0.5, 0.6, 0.7]}

        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        returns_train = train_returns.values.astype(np.float64)
        cash_train = cash_train.values.astype(np.float64)

        val_size = int(0.5 * periods_per_year)
        inner_train = SplitData(signal_train[:-val_size], returns_train[:-val_size], cash_train[:-val_size])
        inner_val = SplitData(signal_train[-val_size:], returns_train[-val_size:], cash_train[-val_size:])

        adapted, sweep_data, shared_metrics = _adaptive_sweep(
            config, inner_train, inner_val, context, adaptive_grid
        )

        assert isinstance(adapted, PipelineConfig)
        assert adapted.vote_threshold in [0.5, 0.6, 0.7]
        assert "hp_names" in sweep_data
        assert "all_results" in sweep_data
        assert "best_combo" in sweep_data
        assert len(sweep_data["all_results"]) > 0
        assert len(shared_metrics) == 3  # (rtrs, cagrs, stabs)


class TestProcessFold:
    def test_first_fold_structure(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(context, 0, folds[0], None)

        assert "oos_returns" in out
        assert "oos_signal" in out
        assert "oos_buy_hold" in out
        assert "oos_dates" in out
        assert "oos_agreement" in out
        assert "oos_deltas" in out
        assert "oos_raw_signals" in out
        assert "oos_folds" in out
        assert "fold_result" in out
        assert "structural_importance" in out
        assert "ensemble_models" in out

    def test_fold_result_keys(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(context, 0, folds[0], None)
        fr = out["fold_result"]

        assert fr["fold"] == 1
        assert "period" in fr
        assert "test_cagr" in fr
        assert "buy_hold_cagr" in fr
        assert "transitions" in fr
        assert "structural_importance" in fr
        assert "duration" in fr
        assert fr["all_failed"] is False

    def test_oos_arrays_consistent_length(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(context, 0, folds[0], None)

        n = len(out["oos_returns"])
        assert n > 0
        assert len(out["oos_signal"]) == n
        assert len(out["oos_buy_hold"]) == n
        assert len(out["oos_dates"]) == n
        assert len(out["oos_agreement"]) == n
        assert out["oos_deltas"].shape[0] == n
        assert out["oos_raw_signals"].shape[0] == n
        assert len(out["oos_folds"]) == n
        assert (out["oos_folds"] == 1).all()

    def test_second_fold_with_prev_sig(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        folds = generate_folds(config.train_start_year, config.first_test_year,
                               config.test_years, pipeline_data["ds"].index[-1].year)
        out1 = _process_fold(context, 0, folds[0], None)
        prev_sig_last = out1["oos_signal"][-1]

        out2 = _process_fold(context, 1, folds[1], prev_sig_last)
        assert out2["fold_result"]["fold"] == 2
        assert (out2["oos_folds"] == 2).all()


class TestRunPipeline:
    def test_full_run(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        result = run_pipeline(
            config, context.signals, context.signal_names,
            context.returns, context.tbill_returns, context.periods_per_year, pipeline_data["ds"].index[-1].year
        )

        assert result is not None
        assert "cagr" in result
        assert "rtr" in result
        assert "max_drawdown" in result
        assert "exposure" in result
        assert "oos_returns" in result
        assert "oos_signal" in result
        assert "oos_dates" in result
        assert "fold_results" in result
        assert "metrics_oos" in result
        assert "metrics_buy_hold_oos" in result
        assert "trader_counts" in result
        assert "last_structural_importance" in result
        assert "fold_structural_importance" in result
        assert "folds" in result
        assert "cagr_thresholds" in result

    def test_fold_count(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        result = run_pipeline(
            config, context.signals, context.signal_names,
            context.returns, context.tbill_returns, context.periods_per_year, pipeline_data["ds"].index[-1].year
        )
        expected_folds = len(generate_folds(
            config.train_start_year, config.first_test_year,
            config.test_years, pipeline_data["ds"].index[-1].year
        ))
        assert len(result["fold_results"]) == expected_folds
        assert len(result["fold_structural_importance"]) == expected_folds

    def test_oos_covers_all_folds(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        result = run_pipeline(
            config, context.signals, context.signal_names,
            context.returns, context.tbill_returns, context.periods_per_year, pipeline_data["ds"].index[-1].year
        )
        unique_folds = set(result["oos_folds"])
        expected = set(range(1, len(result["fold_results"]) + 1))
        assert unique_folds == expected

    def test_metrics_are_finite(self, pipeline_data):
        context = pipeline_data["context"]
        config = context.config
        result = run_pipeline(
            config, context.signals, context.signal_names,
            context.returns, context.tbill_returns, context.periods_per_year, pipeline_data["ds"].index[-1].year
        )
        assert np.isfinite(result["cagr"])
        assert np.isfinite(result["rtr"])
        assert np.isfinite(result["max_drawdown"])
        assert 0 <= result["exposure"] <= 1

    def test_no_folds_returns_none(self, pipeline_data):
        context = pipeline_data["context"]
        no_folds_config = PipelineConfig(
            train_start_year=2020,
            first_test_year=2030,
            test_years=2,
            indicators=context.config.indicators,
        )
        result = run_pipeline(
            no_folds_config, context.signals, context.signal_names,
            context.returns, context.tbill_returns, context.periods_per_year, pipeline_data["ds"].index[-1].year
        )
        assert result is None


def _make_adaptive_context(pipeline_data, config):
    """Build a _Context for a custom config reusing pipeline_data signals."""
    context = pipeline_data["context"]
    trader_configs = make_configs(context.n_signals, config.min_signals_per_trader, config.max_signals_per_trader)
    return _Context(config, context.signals, context.signal_names, context.n_signals,
                    context.returns, context.tbill_returns, trader_configs, context.periods_per_year)


class TestAdaptiveSweepInFold:
    """Test _process_fold with adaptive HP sweep enabled."""

    def test_fold_with_adaptive(self, pipeline_data):
        context = pipeline_data["context"]
        adaptive_config = PipelineConfig(
            train_start_year=1993,
            first_test_year=2005,
            test_years=2,
            trader_min_rtr=0.3,
            vote_threshold=0.6,
            min_signals_per_trader=2,
            max_signals_per_trader=3,
            top_traders=20,
            min_traders_per_group=2,
            max_traders_per_group=2,
            group_min_rtr=0.01,
            group_min_cagr=0.01,
            top_groups=10,
            group_aggregation="equal",
            cagr_thresholds=(0.07, 0.09),
            indicators=context.config.indicators,
            adaptive_val_years=0.5,
            adaptive_grid={"vote_threshold": [0.5, 0.6, 0.7]},
        )
        adaptive_context = _make_adaptive_context(pipeline_data, adaptive_config)
        folds = generate_folds(adaptive_config.train_start_year, adaptive_config.first_test_year,
                               adaptive_config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(adaptive_context, 0, folds[0], None)

        assert out["fold_result"]["all_failed"] is False
        assert "adapted_hps" in out["fold_result"]
        assert "vote_threshold" in out["fold_result"]["adapted_hps"]
        assert out["fold_result"]["sweep_data"] is not None


class TestEdgeCases:
    """Cover remaining edge-case branches in pipeline.py."""

    def test_impossible_cagr_all_none_models(self, pipeline_data):
        """Line 188-189: CAGR threshold so high no trader passes -> model is None."""
        context = pipeline_data["context"]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        folds = generate_folds(context.config.train_start_year, context.config.first_test_year,
                               context.config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        train = SplitData(signal_train, train_returns.values.astype(np.float64),
                          cash_train.values.astype(np.float64))

        overrides = {k: getattr(context.config, k) for k in PipelineConfig.__dataclass_fields__}
        overrides["cagr_thresholds"] = (0.99,)
        impossible_config = PipelineConfig(**overrides)
        models = _build_ensemble_models(impossible_config, train, context)
        assert models == [None]

    def test_all_failed_fold(self, pipeline_data):
        """Line 435: all pipelines fail -> _failed_fold_result path."""
        failed_config = PipelineConfig(
            train_start_year=1993,
            first_test_year=2005,
            test_years=2,
            trader_min_rtr=0.3,
            vote_threshold=0.6,
            min_signals_per_trader=2,
            max_signals_per_trader=3,
            top_traders=20,
            min_traders_per_group=2,
            max_traders_per_group=2,
            group_min_rtr=0.01,
            group_min_cagr=0.01,
            top_groups=10,
            group_aggregation="equal",
            cagr_thresholds=(0.99,),
            indicators=pipeline_data["context"].config.indicators,
        )
        failed_context = _make_adaptive_context(pipeline_data, failed_config)
        folds = generate_folds(failed_config.train_start_year, failed_config.first_test_year,
                               failed_config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(failed_context, 0, folds[0], None)
        assert out["fold_result"]["all_failed"] is True
        assert out["fold_result"]["test_exposure"] == 1.0

    def test_adaptive_grid_rebuilds_trader_configs(self, pipeline_data):
        """Line 168: adaptive_grid with min_signals_per_trader rebuilds configs."""
        context = pipeline_data["context"]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        folds = generate_folds(context.config.train_start_year, context.config.first_test_year,
                               context.config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        train_returns, cash_train = slice_data(context.returns, context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(context.signals, context.signal_names, train_returns.index)
        train = SplitData(signal_train, train_returns.values.astype(np.float64),
                          cash_train.values.astype(np.float64))

        grid = {"min_signals_per_trader": [2, 3]}
        models = _build_ensemble_models(context.config, train, context, adaptive_grid=grid)
        assert isinstance(models, list)
        assert len(models) == len(context.config.cagr_thresholds)

    def test_adaptive_sweep_changes_hp(self, pipeline_data):
        """Line 287: sweep finds better HP -> adapted config differs from base."""
        restrictive_config = PipelineConfig(
            train_start_year=1993,
            first_test_year=2005,
            test_years=2,
            trader_min_rtr=0.3,
            vote_threshold=0.9,  # very restrictive baseline
            min_signals_per_trader=2,
            max_signals_per_trader=3,
            top_traders=20,
            min_traders_per_group=2,
            max_traders_per_group=2,
            group_min_rtr=0.01,
            group_min_cagr=0.01,
            top_groups=10,
            group_aggregation="equal",
            cagr_thresholds=(0.07, 0.09),
            indicators=pipeline_data["context"].config.indicators,
        )
        sweep_context = _make_adaptive_context(pipeline_data, restrictive_config)
        periods_per_year = sweep_context.periods_per_year
        adaptive_grid = {"vote_threshold": [0.3, 0.5, 0.9]}

        folds = generate_folds(restrictive_config.train_start_year, restrictive_config.first_test_year,
                               restrictive_config.test_years, pipeline_data["ds"].index[-1].year)
        train_start, train_end, _, _ = folds[0]
        from spx_ndx.models.spx_consensus.pipeline import slice_data
        train_returns, cash_train = slice_data(sweep_context.returns, sweep_context.tbill_returns, train_start, train_end)
        signal_train = build_sig_matrix(sweep_context.signals, sweep_context.signal_names, train_returns.index)
        returns_train = train_returns.values.astype(np.float64)
        cash_train = cash_train.values.astype(np.float64)

        val_size = int(0.5 * periods_per_year)
        inner_train = SplitData(signal_train[:-val_size], returns_train[:-val_size], cash_train[:-val_size])
        inner_val = SplitData(signal_train[-val_size:], returns_train[-val_size:], cash_train[-val_size:])

        adapted, sweep_data, _ = _adaptive_sweep(restrictive_config, inner_train, inner_val, sweep_context, adaptive_grid)
        # 0.3 or 0.5 should beat 0.9 -> has_changes=True
        assert sweep_data["has_changes"] is True
        assert adapted.vote_threshold != 0.9

    def test_adaptive_val_years_full(self, pipeline_data):
        """Lines 404-405, 417: adaptive_val_years='full' uses train as val and reuses metrics."""
        full_val_config = PipelineConfig(
            train_start_year=1993,
            first_test_year=2005,
            test_years=2,
            trader_min_rtr=0.3,
            vote_threshold=0.6,
            min_signals_per_trader=2,
            max_signals_per_trader=3,
            top_traders=20,
            min_traders_per_group=2,
            max_traders_per_group=2,
            group_min_rtr=0.01,
            group_min_cagr=0.01,
            top_groups=10,
            group_aggregation="equal",
            cagr_thresholds=(0.07, 0.09),
            indicators=pipeline_data["context"].config.indicators,
            adaptive_val_years="full",
            adaptive_grid={"vote_threshold": [0.5, 0.6, 0.7]},
        )
        full_val_context = _make_adaptive_context(pipeline_data, full_val_config)
        folds = generate_folds(full_val_config.train_start_year, full_val_config.first_test_year,
                               full_val_config.test_years, pipeline_data["ds"].index[-1].year)
        out = _process_fold(full_val_context, 0, folds[0], None)
        assert out["fold_result"]["all_failed"] is False
        assert out["fold_result"]["sweep_data"] is not None
