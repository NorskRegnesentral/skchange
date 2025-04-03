"""Tests for MovingWindow and all available scores."""

import numpy as np
import pytest

from skchange.change_detectors import MovingWindow
from skchange.change_scores import CHANGE_SCORES, ContinuousLinearTrendScore
from skchange.change_scores.base import BaseChangeScore
from skchange.costs import COSTS
from skchange.costs.base import BaseCost
from skchange.datasets import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


@pytest.mark.parametrize("ScoreType", SCORES_AND_COSTS)
def test_moving_window_changepoint(ScoreType: type[BaseCost] | type[BaseChangeScore]):
    """Test MovingWindow changepoints."""
    score = ScoreType.create_test_instance()
    if isinstance(score, ContinuousLinearTrendScore):
        pytest.skip(
            "Skipping test for ContinuousLinearTrendCost. It finds two changes in "
            "linear trend (flat, steep, flat), instead of a single change in mean."
        )

    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


def test_moving_window_continuous_linear_trend_score():
    """Test that MovingWindow finds two change points with ContinuousLinearTrendCost."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    score = ContinuousLinearTrendScore.create_test_instance()
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert (
        len(changepoints) == n_segments
        and changepoints[0] == 34
        and changepoints[1] == 65
    )


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
