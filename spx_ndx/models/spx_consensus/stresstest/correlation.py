"""Correlation tests - B&H correlation and baseline signal correlation."""

import numpy as np

from ._common import StressData


def correlation_bh(data: StressData):
    """Correlation between strategy and B&H returns."""
    return {
        "corr_returns": float(np.corrcoef(data.strategy_returns, data.buy_hold_returns)[0, 1]),
        "corr_abs": float(np.corrcoef(np.abs(data.strategy_returns), np.abs(data.buy_hold_returns))[0, 1]),
    }


def baseline_signal_correlation(signal, baseline_signals):
    """Compute correlation between consensus signal and a dict of baseline signals.

    baseline_signals: dict of {name: array} - each array same length as signal.
    Returns dict with per-signal correlations and a full correlation matrix.
    """
    names = list(baseline_signals.keys())
    if not names:
        return {"correlations": {}, "matrix": [], "labels": []}
    arrays = [baseline_signals[n] for n in names]
    all_signals = np.column_stack([signal] + arrays)
    mat = np.corrcoef(all_signals.T)
    labels = ["Consensus"] + names
    correlations = {n: float(mat[0, i + 1]) for i, n in enumerate(names)}
    return {
        "correlations": correlations,
        "matrix": mat.tolist(),
        "labels": labels,
    }
