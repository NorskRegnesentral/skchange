"""Tests for MovingWindow and all available scores."""

import numpy as np
import pytest

from skchange.base import BaseIntervalScorer
from skchange.change_detectors import MovingWindow
from skchange.change_scores import CHANGE_SCORES, ContinuousLinearTrendScore
from skchange.costs import COSTS
from skchange.datasets import generate_alternating_data
from skchange.tests.test_all_interval_scorers import skip_if_no_test_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


@pytest.mark.parametrize("ScoreType", SCORES_AND_COSTS)
def test_moving_window_changepoint(ScoreType: type[BaseIntervalScorer]):
    """Test MovingWindow changepoints."""
    score = ScoreType.create_test_instance()
    skip_if_no_test_data(score)

    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    detector = MovingWindow(score)
    changepoints = detector.fit_predict(df)["ilocs"]
    if isinstance(score, ContinuousLinearTrendScore):
        # ContinuousLinearTrendScore finds two changes in linear trend
        # (flat, steep, flat) instead of a single change in mean.
        assert len(changepoints) == n_segments
    else:
        assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len


def test_moving_window_continuous_linear_trend_score():
    """Test MovingWindow finds two change points with ContinuousLinearTrendScore."""
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
    score = Score.create_test_instance()
    skip_if_no_test_data(score)

    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = MovingWindow(score, penalty=0)
    scores = detector.fit(df).transform_scores(df)
    assert np.all(scores >= 0.0)
    assert len(scores) == len(df)
