"""Tests for MovingWindow and all available scores."""

import numpy as np
import pytest

from skchange.change_detectors import MovingWindow
from skchange.change_scores import CHANGE_SCORES
from skchange.change_scores.base import BaseChangeScore
from skchange.costs import COSTS, ContinuousLinearTrendCost
from skchange.costs.base import BaseCost
from skchange.datasets import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moving_window_changepoint(ScoreType: type[BaseCost] | type[BaseChangeScore]):
    """Test MovingWindow changepoints."""
    score = ScoreType.create_test_instance()
    if isinstance(score, ContinuousLinearTrendCost):
        pytest.skip(
            "Skipping test for ContinuousLinearTrendCost. It fails with MovingWindow."
        )

    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


@pytest.mark.xfail(strict=True)
def test_xfail_moving_window_continuous_linear_trend_cost():
    """Test that MovingWindow fails with ContinuousLinearTrendCost."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    score = ContinuousLinearTrendCost.create_test_instance()
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moving_window_scores(Score):
    """Test MovingWindow scores."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    score = Score.create_test_instance()
    detector = MovingWindow(score)
    scores = detector.fit(df).transform_scores(df)
    assert np.all(scores >= 0.0)
    assert len(scores) == len(df)
