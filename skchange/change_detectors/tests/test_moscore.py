"""Tests for Moscore and all available scores."""

import numpy as np
import pytest

from skchange.change_detectors.moscore import Moscore
from skchange.datasets.generate import generate_teeth_data
from skchange.scores.score_factory import VALID_CHANGE_SCORES


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
def test_moscore_changepoint(score):
    """Test Moscore changepoints."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Moscore(score, fmt="sparse", labels="int_label")
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
def test_moscore_scores(score):
    """Test Moscore scores."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=3
    )
    detector = Moscore(score, fmt="dense", labels="score")
    scores = detector.fit_predict(df)
    assert np.all(scores >= 0.0)


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
def test_moscore_tuning(score):
    """Test Moscore tuning."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = Moscore(score, threshold_scale=None, fmt="dense", labels="indicator")
    detector.fit(df)
    assert detector.threshold_ > 0.0
