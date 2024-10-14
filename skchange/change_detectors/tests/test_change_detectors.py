"""Basic tests for all change detectors."""

import pytest

from skchange.change_detectors import CHANGE_DETECTORS
from skchange.datasets.generate import generate_alternating_data

n_segments = 2
seg_len = 50
changepoint_data = generate_alternating_data(
    n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
)[0]


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_predict(Estimator):
    """Test changepoint detector predict (sparse output)."""
    detector = Estimator.create_test_instance()
    changepoints = detector.fit_predict(changepoint_data)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_transform(Estimator):
    """Test changepoint detector transform (dense output)."""
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(changepoint_data)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_sparse_to_dense(Estimator):
    """Test that predict + sparse_to_dense == transform."""
    detector = Estimator.create_test_instance()
    changepoints = detector.fit_predict(changepoint_data)
    labels = detector.sparse_to_dense(changepoints, changepoint_data.index)
    labels_transform = detector.fit_transform(changepoint_data)
    assert labels.equals(labels_transform)


@pytest.mark.parametrize("Estimator", CHANGE_DETECTORS)
def test_change_detector_dense_to_sparse(Estimator):
    """Test that transform + dense_to_sparse == predict."""
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(changepoint_data)
    changepoints = detector.dense_to_sparse(labels)
    changepoints_predict = detector.fit_predict(changepoint_data)
    assert changepoints.equals(changepoints_predict)
