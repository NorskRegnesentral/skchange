"""Tests for MovingWindow and all available scores."""

import numpy as np
import pytest

from skchange.change_detectors.moscore import MovingWindow
from skchange.change_scores import CHANGE_SCORES
from skchange.costs import COSTS
from skchange.datasets.generate import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moscore_changepoint(Score):
    """Test MovingWindow changepoints."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    detector = MovingWindow(Score())
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moscore_scores(Score):
    """Test MovingWindow scores."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = MovingWindow(Score())
    scores = detector.fit(df).score_transform(df)
    assert np.all(scores >= 0.0)
    assert len(scores) == len(df)


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_moscore_tuning(Score):
    """Test MovingWindow tuning."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = MovingWindow(Score(), threshold_scale=None)
    detector.fit(df)
    assert detector.threshold_ > 0.0
