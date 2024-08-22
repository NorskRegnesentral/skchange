"""Basic tests for all change detectors."""

import pytest

from skchange.change_detectors.moscore import Moscore
from skchange.change_detectors.pelt import Pelt
from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation
from skchange.datasets.generate import generate_teeth_data

change_detectors = [Moscore, Pelt, SeededBinarySegmentation]


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_predict(Estimator):
    """Test changepoint detector predict (sparse output)."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    changepoints = detector.fit_predict(df)
    assert len(changepoints) == n_segments - 1 and changepoints[0] == seg_len - 1


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_transform(Estimator):
    """Test changepoint detector transform (dense output)."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(df)
    assert labels.nunique() == n_segments
    assert labels[seg_len - 1] == 0.0 and labels[seg_len] == 1.0


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_sparse_to_dense(Estimator):
    """Test that predict + sparse_to_dense == transform."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    changepoints = detector.fit_predict(df)
    labels = detector.sparse_to_dense(changepoints, df.index)
    labels_transform = detector.fit_transform(df)
    assert labels.equals(labels_transform)


@pytest.mark.parametrize("Estimator", change_detectors)
def test_change_detector_dense_to_sparse(Estimator):
    """Test that transform + dense_to_sparse == predict."""
    n_segments = 2
    seg_len = 50
    df = generate_teeth_data(
        n_segments=n_segments, mean=10, segment_length=seg_len, p=1, random_state=2
    )
    detector = Estimator.create_test_instance()
    labels = detector.fit_transform(df)
    changepoints = detector.dense_to_sparse(labels)
    changepoints_predict = detector.fit_predict(df)
    assert changepoints.equals(changepoints_predict)
