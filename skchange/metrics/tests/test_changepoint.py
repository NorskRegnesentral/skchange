"""Metric-specific tests for ``skchange.metrics._changepoint``."""

import numpy as np

from skchange.metrics import changepoint_f1_score


def test_f1_score_zero_matches_returns_zero_without_dividing_by_zero():
    """precision + recall == 0 when both arrays are non-empty and no match
    is within ``tolerance``. The guard must return 0.0 instead of raising
    ``ZeroDivisionError``.
    """
    result = changepoint_f1_score(np.array([10]), np.array([100]), tolerance=5)
    assert result == 0.0
