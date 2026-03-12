import numpy as np
import pytest

from spx_ndx.models.spx_consensus.combo import majority_vote, combo_signal


class TestMajorityVote:
    """Tests for majority_vote."""

    def test_all_agree(self):
        """All signals = 1 -> vote = 1."""
        matrix = np.array([[1, 1, 1],
                           [1, 1, 1]], dtype=float)
        result = majority_vote(matrix, min_votes=2)
        np.testing.assert_array_equal(result, [1, 1])

    def test_none_agree(self):
        """All signals = 0 -> vote = 0."""
        matrix = np.array([[0, 0, 0],
                           [0, 0, 0]], dtype=float)
        result = majority_vote(matrix, min_votes=2)
        np.testing.assert_array_equal(result, [0, 0])

    def test_exact_threshold(self):
        """Exactly min_votes signals = 1 -> vote = 1."""
        matrix = np.array([[1, 1, 0],
                           [1, 0, 0]], dtype=float)
        result = majority_vote(matrix, min_votes=2)
        np.testing.assert_array_equal(result, [1, 0])

    def test_single_signal(self):
        """One signal with min_votes=1 -> same as input."""
        matrix = np.array([[1], [0], [1]], dtype=float)
        result = majority_vote(matrix, min_votes=1)
        np.testing.assert_array_equal(result, [1, 0, 1])

    def test_returns_float_array(self):
        """Result should be float array of 0/1."""
        matrix = np.array([[1, 0], [0, 1]], dtype=float)
        result = majority_vote(matrix, min_votes=1)
        assert result.dtype == np.float64


class TestComboSignal:
    """Tests for combo_signal - select columns + vote."""

    def test_basic(self):
        """Select 2 of 3 signals, vote with min_votes=2."""
        # 3 signals, 4 time steps
        matrix = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ], dtype=float)
        combo = (0, 1)
        result = combo_signal(matrix, combo, min_votes=2)
        # t0: sig0=1, sig1=1 -> 2>=2 -> 1
        # t1: sig0=1, sig1=0 -> 1<2  -> 0
        # t2: sig0=0, sig1=0 -> 0<2  -> 0
        # t3: sig0=1, sig1=1 -> 2>=2 -> 1
        np.testing.assert_array_equal(result, [1, 0, 0, 1])

    def test_min_votes_1(self):
        """min_votes=1 -> OR logic."""
        matrix = np.array([
            [1, 0],
            [0, 1],
            [0, 0],
        ], dtype=float)
        result = combo_signal(matrix, (0, 1), min_votes=1)
        np.testing.assert_array_equal(result, [1, 1, 0])

    def test_returns_new_array(self):
        """Must not modify input matrix."""
        matrix = np.array([[1, 0], [0, 1]], dtype=float)
        original = matrix.copy()
        combo_signal(matrix, (0, 1), min_votes=1)
        np.testing.assert_array_equal(matrix, original)
