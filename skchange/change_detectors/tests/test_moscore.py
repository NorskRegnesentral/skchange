"""Tests for Moscore and all available scores."""

import numpy as np
import pytest

from skchange.change_detectors.moscore import Moscore
from skchange.change_scores import CHANGE_SCORES
from skchange.datasets.generate import generate_alternating_data


@pytest.mark.parametrize("Score", CHANGE_SCORES)
def test_moscore_changepoint(Score):
    """Test Moscore changepoints."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=15, segment_length=seg_len, p=1, random_state=2
    )
    detector = Moscore(Score())
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Score", CHANGE_SCORES)
def test_moscore_scores(Score):
    """Test Moscore scores."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = Moscore(Score())
    scores = detector.fit(df).score_transform(df)
    assert np.all(scores >= 0.0)
    assert len(scores) == len(df)


@pytest.mark.parametrize("Score", CHANGE_SCORES)
def test_moscore_tuning(Score):
    """Test Moscore tuning."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = Moscore(Score(), threshold_scale=None)
    detector.fit(df)
    assert detector.threshold_ > 0.0
