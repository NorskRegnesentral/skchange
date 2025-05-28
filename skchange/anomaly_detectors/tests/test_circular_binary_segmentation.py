"""Tests for CircularBinarySegmentation."""

import pytest

from skchange.anomaly_detectors import CircularBinarySegmentation
from skchange.change_scores import ChangeScore
from skchange.costs import COSTS


def test_invalid_change_scores():
    """
    Test that CircularBinarySegmentation raises an error when given an invalid score.
    """
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation("l2")
    with pytest.raises(ValueError, match="anomaly_score"):
        CircularBinarySegmentation(ChangeScore(COSTS[2].create_test_instance()))
