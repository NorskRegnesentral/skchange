"""Tests for CircularBinarySegmentation."""

import pytest

from skchange.anomaly_detectors import CircularBinarySegmentation
from skchange.change_scores import ChangeScore
from skchange.costs import COSTS, L2Cost
from skchange.datasets import generate_alternating_data


def test_invalid_change_scores():
    """
    Test that CircularBinarySegmentation raises an error when given an invalid score.
    """
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation("l2")
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation(ChangeScore(COSTS[2].create_test_instance()))


class WrongTupleMinSizeL2Cost(L2Cost):
    """A custom cost class that changes `min_size` to return a tuple."""

    @property
    def min_size(self) -> tuple[int, int]:
        """Return a tuple instead of an integer."""
        return (3, 3)


class CorrectTupleMinSizeL2Cost(L2Cost):
    """A custom cost class that changes `min_size` to return a tuple."""

    @property
    def min_size(self) -> tuple[int, ...]:
        """Return a tuple instead of an integer."""
        return (1,)


def test_CircularBinarySegmentation_init_with_wrong_tuple_min_size_raises():
    """Test anomaly detector with a custom cost class that returns a tuple."""
    data = generate_alternating_data(
        n_segments=3, mean=10, segment_length=50, p=1, random_state=2
    )
    cpd = CircularBinarySegmentation(
        anomaly_score=WrongTupleMinSizeL2Cost(param=0.0),
        min_segment_length=2,
        max_interval_length=10,
    )

    cpd.fit(X=data)
    with pytest.raises(
        ValueError,
        match=(
            "The `min_size` must be a scalar or a tuple with one "
            "less entry than the number number of columns in `cuts`."
        ),
    ):
        cpd.predict(data)


def test_CircularBinarySegmentation_init_with_correct_tuple_min_size():
    """Test anomaly detector with a custom cost class that returns a tuple."""
    data = generate_alternating_data(
        n_segments=3, mean=10, segment_length=50, p=1, random_state=2
    )
    cpd = CircularBinarySegmentation(
        anomaly_score=CorrectTupleMinSizeL2Cost(param=0.0),
        min_segment_length=2,
        max_interval_length=10,
    )

    cpd.fit(X=data)
    cpd.predict(data)
