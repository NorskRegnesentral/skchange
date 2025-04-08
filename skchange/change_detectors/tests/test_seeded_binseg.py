"""Tests for MovingWindow and all available scores."""

import pytest

from skchange.change_detectors import SeededBinarySegmentation
from skchange.change_scores import CHANGE_SCORES
from skchange.costs import COSTS
from skchange.datasets import generate_alternating_data

SCORES_AND_COSTS = CHANGE_SCORES + COSTS


def test_invalid_parameters():
    """Test invalid input parameters to SeededBinarySegmentation.

    These tests serve as tests for the input validators.
    """
    with pytest.raises(ValueError):
        SeededBinarySegmentation(penalty=-0.1)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=1.0)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=None)
    with pytest.raises(ValueError):
        SeededBinarySegmentation(growth_factor=2.01)


@pytest.mark.parametrize("selection_method", ["greedy", "narrowest"])
def test_selection_method(selection_method):
    """Test SeededBinarySegmentation selection method."""
    n_segments = 2
    seg_len = 10
    df = generate_alternating_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=200
    )
    detector = SeededBinarySegmentation.create_test_instance()
    detector.set_params(selection_method=selection_method)
    changepoints = detector.fit_predict(df)["ilocs"]
    assert len(changepoints) == n_segments - 1
    assert changepoints[0] == 10


def test_invalid_selection_method():
    """Test invalid selection method."""
    detector = SeededBinarySegmentation.create_test_instance()
    with pytest.raises(ValueError):
        detector.set_params(selection_method="greedy2")
