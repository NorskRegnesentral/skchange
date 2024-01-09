"""Tests for MoscoreAnomaly and all available anomaly scores."""

import numpy as np
import pytest

from skchange.anomaly_detectors.moscore_anomaly import MoscoreAnomaly
from skchange.datasets.generate import generate_anomalous_data
from skchange.scores.score_factory import VALID_ANOMALY_SCORES

true_anomalies = [(50, 59), (120, 129)]
anomaly_data = generate_anomalous_data(
    200, anomalies=true_anomalies, means=[10.0, 5.0], random_state=5
)


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_anomalies(score):
    """Test Moscore anomalies."""
    detector = MoscoreAnomaly.create_test_instance()
    detector.set_params(score=score, fmt="sparse", labels="int_label")
    anomalies = detector.fit_predict(anomaly_data)
    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.loc[i, "start"] == start and anomalies.loc[i, "end"] == end


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_scores(score):
    """Test Moscore scores."""
    detector = MoscoreAnomaly.create_test_instance()
    detector.set_params(score=score, fmt="sparse", labels="int_label")
    scores = detector.fit_predict(anomaly_data)
    assert np.all(scores >= 0.0)


@pytest.mark.parametrize("score", VALID_ANOMALY_SCORES)
def test_moscore_tuning(score):
    """Test Moscore tuning."""
    detector = MoscoreAnomaly.create_test_instance()
    detector.set_params(
        score=score, threshold_scale=None, fmt="dense", labels="indicator"
    )
    detector.fit(anomaly_data)
    assert detector.threshold_ > 0.0
