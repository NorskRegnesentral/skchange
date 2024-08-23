"""Basic tests for all anomaly detectors."""

import pandas as pd
import pytest

from skchange.anomaly_detectors import ANOMALY_DETECTORS, COLLECTIVE_ANOMALY_DETECTORS
from skchange.anomaly_detectors.base import CollectiveAnomalyDetector
from skchange.datasets.generate import generate_anomalous_data

true_anomalies = [(30, 34), (70, 75)]
anomaly_data = generate_anomalous_data(
    100, anomalies=true_anomalies, means=[10.0, 15.0], random_state=2
)


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_predict(Estimator):
    """Test collective anomaly detector's predict method (sparse output)."""
    detector = Estimator.create_test_instance()
    anomalies = detector.fit_predict(anomaly_data)
    if isinstance(anomalies, pd.DataFrame):
        anomalies = anomalies.iloc[:, 0]

    assert len(anomalies) == len(true_anomalies)
    for i, (start, end) in enumerate(true_anomalies):
        assert anomalies.array.left[i] == start and anomalies.array.right[i] == end


@pytest.mark.parametrize("Estimator", COLLECTIVE_ANOMALY_DETECTORS)
def test_collective_anomaly_detector_transform(Estimator):
    """Test collective anomaly detector's transform method (dense output)."""
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(anomaly_data)
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]

    true_collective_anomalies = pd.IntervalIndex.from_tuples(
        true_anomalies, closed="both"
    )
    true_anomaly_labels = CollectiveAnomalyDetector.sparse_to_dense(
        true_collective_anomalies, anomaly_data.index
    )
    labels.equals(true_anomaly_labels)

    # Similar test that does not depend on sparse_to_dense, just to be sure.
    assert labels.nunique() == len(true_anomalies) + 1
    for i, (start, end) in enumerate(true_anomalies):
        assert (labels.iloc[start : end + 1] == i + 1).all()


@pytest.mark.parametrize("Estimator", ANOMALY_DETECTORS)
def test_anomaly_detector_sparse_to_dense(Estimator):
    """Test that predict + sparse_to_dense == transform."""
    detector = Estimator.create_test_instance()
    anomalies = detector.fit_predict(anomaly_data)
    labels_predict_convert = detector.sparse_to_dense(
        anomalies, anomaly_data.index, anomaly_data.columns
    )
    if isinstance(labels_predict_convert, pd.Series):
        # transforms does output conversion to match the input. This is not required of
        # spare_to_dense.
        labels_predict_convert = labels_predict_convert.to_frame()
    labels_transform = detector.fit_transform(anomaly_data)
    assert labels_predict_convert.equals(labels_transform)


@pytest.mark.parametrize("Estimator", ANOMALY_DETECTORS)
def test_anomaly_detector_dense_to_sparse(Estimator):
    """Test that transform + dense_to_sparse == predict."""
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(anomaly_data)
    anomalies_transform_convert = detector.dense_to_sparse(labels)
    anomalies_predict = detector.fit_predict(anomaly_data)
    assert anomalies_transform_convert.equals(anomalies_predict)
