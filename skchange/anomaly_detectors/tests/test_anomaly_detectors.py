"""Basic tests for all anomaly detectors."""

import pandas as pd
import pytest

from skchange.anomaly_detectors import COLLECTIVE_ANOMALY_DETECTORS
from skchange.anomaly_detectors.base import BaseCollectiveAnomalyDetector
from skchange.datasets.generate import generate_anomalous_data

true_anomalies = [(30, 35), (70, 75)]
anomaly_data = generate_anomalous_data(
    100, anomalies=true_anomalies, means=[10.0, 15.0], random_state=2
)
anomaly_free_data = generate_anomalous_data(100, random_state=1)


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_predict(Estimator: BaseCollectiveAnomalyDetector):
    """Test collective anomaly detector's predict method (sparse output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    anomalies = detector.predict(anomaly_data)["ilocs"]

    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.array.left[i] == start and anomalies.array.right[i] == end


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_transform(
    Estimator: BaseCollectiveAnomalyDetector,
):
    """Test collective anomaly detector's transform method (dense output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    labels = detector.transform(anomaly_data)
    true_collective_anomalies = pd.DataFrame(
        {"ilocs": pd.IntervalIndex.from_tuples(true_anomalies, closed="left")}
    )
    true_anomaly_labels = BaseCollectiveAnomalyDetector.sparse_to_dense(
        true_collective_anomalies, anomaly_data.index
    )
    labels.equals(true_anomaly_labels)

    # Similar test that does not depend on sparse_to_dense, just to be sure.
    labels = labels.iloc[:, 0]
    assert labels.nunique() == len(true_anomalies) + 1
    for i, (start, end) in enumerate(true_anomalies):
        assert (labels.iloc[start:end] == i + 1).all()
