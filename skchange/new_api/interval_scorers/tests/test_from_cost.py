"""Tests for interval scorers constructed from costs."""

from unittest.mock import Mock

import numpy as np

from skchange.new_api.interval_scorers import CostChangeScore, L2Cost


def test_cost_change_score_deduplicates_all_cost_intervals():
    """Left, right, and full intervals are deduplicated in one cost call."""
    X = np.arange(6.0).reshape(-1, 1)
    scorer = CostChangeScore(L2Cost()).fit(X)
    cache = scorer.precompute(X)
    interval_specs = np.array([[0, 2, 4], [0, 4, 6]])
    expected_unique_intervals = np.array([[0, 2], [0, 4], [0, 6], [2, 4], [4, 6]])

    scorer.cost_.evaluate = Mock(wraps=scorer.cost_.evaluate)
    scorer.evaluate(cache, interval_specs)

    scorer.cost_.evaluate.assert_called_once()
    evaluated_intervals = scorer.cost_.evaluate.call_args.args[1]
    np.testing.assert_array_equal(evaluated_intervals, expected_unique_intervals)
