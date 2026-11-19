"""Tests for the cumulative-sum ContinuousLinearTrendScore against references."""

import numpy as np
import pytest

from skchange.interval_scorers import ContinuousLinearTrendScore
from skchange.interval_scorers._change_scores.continuous_linear_trend_score import (
    _analytical_cont_piecewise_linear_trend_score,
    _lin_reg_cont_piecewise_linear_trend_score,
)


def _all_splits_specs(intervals, min_size=1):
    """Expand ``(start, end)`` intervals into rows ``[start, split, end]``."""
    return np.array(
        [[s, b, e] for s, e in intervals for b in range(s + min_size, e - min_size)],
        dtype=np.int64,
    )


def test_matches_analytical_regular_times_multivariate():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)).cumsum(axis=0)
    specs = _all_splits_specs([(0, 300), (20, 120), (150, 153), (200, 300)])

    scorer = ContinuousLinearTrendScore().fit(X)
    scores = scorer.evaluate(scorer.precompute(X), specs)
    expected = _analytical_cont_piecewise_linear_trend_score(
        specs[:, 0], specs[:, 1], specs[:, 2], X
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-8, atol=1e-8)


def test_matches_lstsq_irregular_times():
    rng = np.random.default_rng(1)
    n = 200
    times = np.cumsum(rng.uniform(0.2, 3.0, size=n))
    X = np.column_stack([times, rng.normal(size=(n, 2)).cumsum(axis=0)])
    specs = _all_splits_specs([(0, n), (30, 90), (100, 104)])

    scorer = ContinuousLinearTrendScore(time_col=0).fit(X)
    scores = scorer.evaluate(scorer.precompute(X), specs)
    expected = _lin_reg_cont_piecewise_linear_trend_score(
        specs[:, 0], specs[:, 1], specs[:, 2], X[:, 1:], times - times[0]
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-8)


def test_short_interval_far_from_time_origin():
    """Per-interval sums must not lose precision at large time offsets."""
    rng = np.random.default_rng(2)
    n = 1_000_000
    X = (0.001 * np.arange(n) + rng.normal(size=n))[:, None]
    specs = _all_splits_specs([(n - 200, n)])

    scorer = ContinuousLinearTrendScore().fit(X)
    scores = scorer.evaluate(scorer.precompute(X), specs)
    expected = _analytical_cont_piecewise_linear_trend_score(
        specs[:, 0], specs[:, 1], specs[:, 2], X
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-6, atol=1e-8)


def test_unordered_rows_and_runs_of_length_one():
    """Row order, and rows that do not share an interval, do not change scores."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 2))
    specs = _all_splits_specs([(0, 120), (10, 70), (40, 100)])
    shuffled = specs[rng.permutation(len(specs))]

    scorer = ContinuousLinearTrendScore().fit(X)
    cache = scorer.precompute(X)
    expected = _analytical_cont_piecewise_linear_trend_score(
        shuffled[:, 0], shuffled[:, 1], shuffled[:, 2], X
    )
    np.testing.assert_allclose(
        scorer.evaluate(cache, shuffled), expected, rtol=1e-8, atol=1e-8
    )


@pytest.mark.parametrize("length", [3, 4])
def test_minimum_length_intervals(length):
    X = np.random.default_rng(4).normal(size=(10, 1))
    specs = _all_splits_specs([(2, 2 + length)])

    scorer = ContinuousLinearTrendScore().fit(X)
    scores = scorer.evaluate(scorer.precompute(X), specs)
    expected = _analytical_cont_piecewise_linear_trend_score(
        specs[:, 0], specs[:, 1], specs[:, 2], X
    )
    np.testing.assert_allclose(scores, expected, rtol=1e-8, atol=1e-10)
