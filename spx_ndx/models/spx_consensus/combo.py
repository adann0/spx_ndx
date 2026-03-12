"""Combination and voting logic for trader/group signals."""

import numpy as np


def majority_vote(signal_matrix, min_votes):
    """Apply majority vote across columns of a signal matrix.

    Returns 1D float array of length T with 0/1 values.
    """
    total = signal_matrix.sum(axis=1)
    return (total >= min_votes).astype(np.float64)


def combo_signal(signal_matrix, combo, min_votes):
    """Build a combined signal from selected columns + vote.

    Returns 1D float array of length T with 0/1 values.
    """
    selected = signal_matrix[:, combo]
    return majority_vote(selected, min_votes)
