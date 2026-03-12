"""Stress test suite - one module per test."""

from ._common import StressData, compute_baseline, find_drawdowns
from .permutation import permutation_test
from .bootstrap import bootstrap_returns
from .transaction_costs import txcosts_sensitivity
from .signal_noise import signal_noise_injection
from .return_noise import return_noise_injection
from .montecarlo import montecarlo_paths
from .regime import regime_split
from .cross_index import cross_index
from .vintage import vintage_analysis
from .drawdown import drawdown_analysis
from .rolling_alpha import rolling_alpha
from .block_bootstrap import block_bootstrap
from .correlation import correlation_bh, baseline_signal_correlation
from .decade import decade_split
from .rolling_rtr import rolling_rtr_sharpe

__all__ = [
    "StressData", "compute_baseline", "find_drawdowns",
    "permutation_test", "bootstrap_returns", "txcosts_sensitivity",
    "signal_noise_injection", "return_noise_injection", "montecarlo_paths",
    "regime_split", "cross_index", "vintage_analysis", "drawdown_analysis",
    "rolling_alpha", "block_bootstrap",
    "correlation_bh", "baseline_signal_correlation",
    "decade_split", "rolling_rtr_sharpe",
]
