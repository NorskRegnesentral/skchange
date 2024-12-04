"""Basic tests for all anomaly detectors."""

import pandas as pd
import pytest

from skchange.anomaly_detectors import (
    ANOMALY_DETECTORS,
    COLLECTIVE_ANOMALY_DETECTORS,
)
from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.datasets.generate import generate_anomalous_data

true_anomalies = [(30, 35), (70, 75)]
anomaly_data = generate_anomalous_data(
    100, anomalies=true_anomalies, means=[10.0, 15.0], random_state=2
)
anomaly_free_data = generate_anomalous_data(100, random_state=1)


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_predict(Estimator: CollectiveAnomalyDetector):
    """Test collective anomaly detector's predict method (sparse output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    anomalies = detector.predict(anomaly_data)["ilocs"]

    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.array.left[i] == start and anomalies.array.right[i] == end


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_transform(Estimator: CollectiveAnomalyDetector):
    """Test collective anomaly detector's transform method (dense output)."""
    detector = Estimator.create_test_instance()
    detector.fit(anomaly_free_data)
    labels = detector.transform(anomaly_data)
    # if isinstance(labels, pd.DataFrame):
    #     labels = labels.iloc[:, 0]

    true_collective_anomalies = pd.DataFrame(
        {"ilocs": pd.IntervalIndex.from_tuples(true_anomalies, closed="left")}
    )
    true_anomaly_labels = CollectiveAnomalyDetector.sparse_to_dense(
        true_collective_anomalies, anomaly_data.index
    )
    labels.equals(true_anomaly_labels)

    # Similar test that does not depend on sparse_to_dense, just to be sure.
    labels = labels.iloc[:, 0]
    assert labels.nunique() == len(true_anomalies) + 1
    for i, (start, end) in enumerate(true_anomalies):
        assert (labels.iloc[start:end] == i + 1).all()


@pytest.mark.parametrize("Estimator", ANOMALY_DETECTORS)
def test_anomaly_detector_sparse_to_dense(Estimator: CollectiveAnomalyDetector):
    """Test that predict + sparse_to_dense == transform."""
    detector = Estimator.create_test_instance()
    anomalies = detector.fit_predict(anomaly_data)
    labels_predict_convert = detector.sparse_to_dense(
        anomalies, anomaly_data.index, anomaly_data.columns
    )
    labels_transform = detector.fit_transform(anomaly_data)
    assert labels_predict_convert.equals(labels_transform)


@pytest.mark.parametrize("Estimator", ANOMALY_DETECTORS)
def test_anomaly_detector_dense_to_sparse(Estimator: CollectiveAnomalyDetector):
    """Test that transform + dense_to_sparse == predict."""
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(anomaly_data)
    anomalies_transform_convert = detector.dense_to_sparse(labels)
    anomalies_predict = detector.fit_predict(anomaly_data)
    assert anomalies_transform_convert.equals(anomalies_predict)
