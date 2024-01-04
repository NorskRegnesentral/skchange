"""Tests for Moscore and all available scores."""

import numpy as np
import pandas as pd
import pytest

from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation
from skchange.datasets.generate import generate_teeth_data
from skchange.scores.score_factory import VALID_CHANGE_SCORES


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


@pytest.mark.parametrize("score", VALID_CHANGE_SCORES)
def test_binseg_tuning(score):
    """Test SeededBinarySegmentation tuning."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=4
    )
    detector = SeededBinarySegmentation(
        score, threshold_scale=None, fmt="dense", labels="score"
    )
    detector.fit(df)
    scores = detector.predict(df)
    assert detector.threshold_ >= scores["score"].mean()
    assert detector.threshold_ <= scores["score"].max()
