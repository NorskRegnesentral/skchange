"""Tests for MoscoreAnomaly and all available anomaly scores."""

import numpy as np
import pytest

from skchange.anomaly_detectors.moscore_anomaly import MoscoreAnomaly
from skchange.datasets.generate import generate_anomalous_data
from skchange.scores.score_factory import VALID_ANOMALY_SCORES

true_anomalies = [(30, 34), (70, 75)]
anomaly_data = generate_anomalous_data(
    100, anomalies=true_anomalies, means=[10.0, 15.0], random_state=103
)


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_anomalies(score):
    """Test Moscore anomalies."""
    detector = MoscoreAnomaly(
        score, min_anomaly_length=4, max_anomaly_length=10, left_bandwidth=20
    )
    detector.set_params(score=score)
    anomalies = detector.fit_predict(anomaly_data)
    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.array.left[i] == start and anomalies.array.right[i] == end


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_scores(score):
    """Test MoscoreAnomaly scores."""
    detector = MoscoreAnomaly.create_test_instance()
    detector.set_params(score=score)
    detector.fit_predict(anomaly_data)
    assert np.all(detector.scores >= 0.0)


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_tuning(score):
    """Test Moscore tuning."""
    detector = MoscoreAnomaly.create_test_instance()
    detector.set_params(score=score, threshold_scale=None)
    detector.fit(anomaly_data)
    assert detector.threshold_ > 0.0
