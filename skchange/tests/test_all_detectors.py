"""Tests for all annotators/detectors in skchange."""

import numpy as np
import pandas as pd
import pytest

from skchange.anomaly_detectors import ANOMALY_DETECTORS
from skchange.base import BaseDetector
from skchange.change_detectors import CHANGE_DETECTORS
from skchange.datasets.generate import generate_anomalous_data

ALL_DETECTORS = ANOMALY_DETECTORS + CHANGE_DETECTORS


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_fit(Detector: BaseDetector):
    """Test fit method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    y = pd.Series(np.zeros(len(x)))  # For coverage testing.
    fit_detector = detector.fit(x, y)
    assert issubclass(detector.__class__, BaseDetector)
    assert issubclass(fit_detector.__class__, BaseDetector)
    assert isinstance(fit_detector, Detector)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_predict(Detector: BaseDetector):
    """Test predict method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=60)
    y = detector.fit_predict(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_transform(Detector: BaseDetector):
    """Test transform method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=61)
    y = detector.fit_transform(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))
    assert len(x) == len(y)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_transform_scores(Detector: BaseDetector):
    """Test transform_scores method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=62)
    try:
        y = detector.fit(x).transform_scores(x)
        assert isinstance(y, (pd.Series, pd.DataFrame))
    except NotImplementedError:
        pass


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_update(Detector: BaseDetector):
    """Test update method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    x_train = x.iloc[:20]
    x_next = x[20:]
    detector.fit(x_train)
    detector.update_predict(x_next)
    assert issubclass(detector.__class__, BaseDetector)
    assert isinstance(detector, Detector)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_sparse_to_dense(Detector):
    """Test that predict + sparse_to_dense == transform."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=63)
    detections = detector.fit_predict(x)
    labels = detector.sparse_to_dense(detections, x.index, x.columns)
    labels_transform = detector.fit_transform(x)
    assert labels.equals(labels_transform)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_dense_to_sparse(Detector):
    """Test that transform + dense_to_sparse == predict."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=63)
    labels = detector.fit_transform(x)
    detections = detector.dense_to_sparse(labels)
    detections_predict = detector.fit_predict(x)
    assert detections.equals(detections_predict)


def test_detector_not_implemented_methods():
    detector = BaseDetector()
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    with pytest.raises(NotImplementedError):
        detector.fit(x)

    detector._is_fitted = True  # Required for the following functions to run
    with pytest.raises(NotImplementedError):
        detector.predict(x)
    with pytest.raises(NotImplementedError):
        detector.transform(x)
    with pytest.raises(NotImplementedError):
        detector.transform_scores(x)
