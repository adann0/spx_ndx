"""Orchestrator: call all plot functions and return saved paths."""

import sys

import pandas as pd

from .permutation import plot_permutation
from .bootstrap import plot_bootstrap
from .txcosts import plot_txcosts
from .signal_noise import plot_signal_noise
from .return_noise import plot_return_noise

from .regimes import plot_regimes
from .cross_index import plot_cross_index
from .vintage import plot_vintage
from .vintage_dca import plot_vintage_dca
from .drawdown_scatter import plot_drawdown_scatter
from .cumulative_alpha import plot_cumulative_alpha
from .summary_table import plot_summary_table
from .explain import plot_explain
from .proximity import (
    plot_proximity, plot_proximity_gauges, plot_proximity_heatmap,
    plot_proximity_radar, plot_proximity_bubble, plot_proximity_thermo,
)

from .folds_cagr import plot_folds_cagr
from .decades import plot_decades
from .rolling_rtr import plot_rolling_rtr


def generate_all(metrics, df, label,
                 explain_path="output/spx_consensus_explainability.json"):
    """Call all plot functions and return list of saved paths."""
    signal = df["signal"].values
    strategy_returns = df["strategy_returns"].values
    buy_hold_returns = df["buy_hold_returns"].values
    cash_returns = df["cash_returns"].values

    frequency = metrics["meta"].get("frequency", "monthly")
    _dataset_cache = [None]

    def _get_dataset():
        if _dataset_cache[0] is None:
            _dataset_cache[0] = pd.read_parquet(f"datas/dataset_{frequency}.parquet")
            _dataset_cache[0].index = pd.to_datetime(_dataset_cache[0].index)
        return _dataset_cache[0]

    saved = []

    def _add(result):
        if result is None:
            return
        if isinstance(result, list):
            saved.extend([p for p in result if p])
        elif isinstance(result, tuple):
            _, p = result
            if isinstance(p, list):
                saved.extend([x for x in p if x])
            elif p:
                saved.append(p)

    dataframe, label_arg = df, label
    explain_path_arg = explain_path
    _NEEDS_DATASET = "dataset"  # marker for lazy dataset loading
    registry = [
        # ── Significatif ? (1-2) ──
        (plot_permutation, (metrics, dataframe, label_arg)),                                          # 1
        (plot_bootstrap, (metrics, dataframe, label_arg)),                                            # 2
        # ── Robuste ? (3-5) ──
        (plot_signal_noise, (metrics, dataframe, label_arg)),                                         # 3
        (plot_return_noise, (metrics, dataframe, label_arg)),                                         # 4
        (plot_txcosts, (metrics, dataframe, label_arg)),                                              # 5
        # ── Stable dans le temps ? (6-10) ──
        (plot_vintage, (metrics, dataframe, label_arg)),                                              # 6
        (plot_vintage_dca, (metrics, dataframe, label_arg, signal, buy_hold_returns, cash_returns),
         {_NEEDS_DATASET: True}),                                                                     # 6b
        (plot_decades, (metrics, dataframe, label_arg)),                                              # 7
        (plot_regimes, (metrics, dataframe, label_arg)),                                              # 8
        (plot_drawdown_scatter, (metrics, dataframe, label_arg)),                                     # 9
        (plot_rolling_rtr, (metrics, dataframe, label_arg)),                                          # 10
        (plot_cumulative_alpha, (metrics, dataframe, label_arg, strategy_returns, buy_hold_returns)),  # 11
        # ── Non-numerotes ──
        (plot_cross_index, (metrics, dataframe, label_arg, signal, strategy_returns, buy_hold_returns, cash_returns),
         {_NEEDS_DATASET: True}),
        (plot_summary_table, (metrics, dataframe, label_arg)),
        (plot_explain, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity_gauges, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity_heatmap, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity_radar, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity_bubble, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_proximity_thermo, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg, _NEEDS_DATASET: True}),
        (plot_folds_cagr, (metrics, dataframe, label_arg), {"explain_path": explain_path_arg}),
    ]

    for entry in registry:
        fn, args = entry[0], entry[1]
        kwargs = dict(entry[2]) if len(entry) > 2 else {}
        needs_ds = kwargs.pop(_NEEDS_DATASET, False)
        try:
            if needs_ds:
                kwargs["dataset"] = _get_dataset()
            _add(fn(*args, **kwargs))
        except Exception as exc:
            print(f"  [SKIP] {fn.__name__}: {exc}", file=sys.stderr)

    return saved
