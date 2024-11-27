"""Tests for all annotators/detectors in skchange."""

import numpy as np
import pandas as pd
import pytest
from sktime.utils._testing.annotation import make_annotation_problem
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from skchange.anomaly_detectors import ANOMALY_DETECTORS
from skchange.base import BaseDetector
from skchange.change_detectors import CHANGE_DETECTORS
from skchange.datasets.generate import generate_anomalous_data

ALL_DETECTORS = ANOMALY_DETECTORS + CHANGE_DETECTORS


@parametrize_with_checks(ALL_DETECTORS)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


@pytest.mark.parametrize("Detector", ALL_DETECTORS)
def test_detector_fit(Detector: BaseDetector):
    """Test fit method output."""
    detector = Detector.create_test_instance()
    x = make_annotation_problem(n_timepoints=50, estimator_type="None")
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
    x = make_annotation_problem(n_timepoints=30, estimator_type="None")
    x_train = x[:20].to_frame()
    x_next = x[20:].to_frame()
    detector.fit(x_train)
    detector.update_predict(x_next)
    assert issubclass(detector.__class__, BaseDetector)
    assert isinstance(detector, Detector)


def test_detector_not_implemented_methods():
    detector = BaseDetector()
    x = make_annotation_problem(n_timepoints=20, estimator_type="None")
    with pytest.raises(NotImplementedError):
        detector.fit(x)

    detector._is_fitted = True  # Required for the following functions to run
    with pytest.raises(NotImplementedError):
        detector.predict(x)
    with pytest.raises(NotImplementedError):
        detector.transform(x)
    with pytest.raises(NotImplementedError):
        detector.transform_scores(x)
    with pytest.raises(NotImplementedError):
        detector.dense_to_sparse(x)
    with pytest.raises(NotImplementedError):
        detector.sparse_to_dense(x, x.index, pd.Index(["a"]))
