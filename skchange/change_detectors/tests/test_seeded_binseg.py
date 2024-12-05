"""Tests for MovingWindow and all available scores."""

import numpy as np
import pandas as pd
import pytest

from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation
from skchange.change_scores import CHANGE_SCORES
from skchange.costs import COSTS
from skchange.datasets.generate import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


def test_invalid_parameters():
    """Test invalid input parameters to SeededBinarySegmentation.

    These tests serve as tests for the input validators.
    """
    with pytest.raises(ValueError):
        SeededBinarySegmentation(threshold_scale=-0.1)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(min_segment_length=0)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(min_segment_length=None)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(min_segment_length=5, max_interval_length=9)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=1.0)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=None)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=2.01)


def test_invalid_data():
    """Test invalid input data to SeededBinarySegmentation.

    These tests serve as tests for the input validators.
    """
    detector = SeededBinarySegmentation()
    with pytest.raises(ValueError):
        detector.fit(np.array([1.0]))

    with pytest.raises(ValueError):
        detector.fit(pd.Series([1.0, np.nan, 1.0, 1.0]))


@pytest.mark.parametrize("Score", SCORES_AND_COSTS)
def test_binseg_tuning(Score):
    """Test SeededBinarySegmentation tuning."""
    n_segments = 2
    seg_len = 50
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    score = Score.create_test_instance()
    detector = SeededBinarySegmentation(score, threshold_scale=None)
    detector.fit_predict(df)
    assert detector.threshold_ >= detector.scores["score"].mean()
    assert detector.threshold_ <= detector.scores["score"].max()


@pytest.mark.parametrize("min_segment_length", range(1, 5))
def test_min_segment_length(min_segment_length):
    """Test SeededBinarySegmentation min_segment_length."""
    n_segments = 1
    seg_len = 10
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = SeededBinarySegmentation.create_test_instance()
    detector.set_params(min_segment_length=min_segment_length, threshold_scale=0.0)
    changepoints = detector.fit_predict(df)["ilocs"]
    changepoints = np.concatenate([[0], changepoints, [len(df)]])
    assert np.all(np.diff(changepoints) >= min_segment_length)
