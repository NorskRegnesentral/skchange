"""Tests for all annotators/detectors in skchange."""

import pandas as pd
import pytest
from sktime.utils._testing.annotation import make_annotation_problem
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from skchange.base import BaseDetector
from skchange.change_detectors.tests.test_change_detectors import change_detectors
from skchange.datasets.generate import generate_anomalous_data

# all_detectors = anomaly_detectors + change_detectors
all_detectors = change_detectors


@parametrize_with_checks(all_detectors)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


@pytest.mark.parametrize("Detector", all_detectors)
def test_detector_fit(Detector):
    """Test fit method output."""
    detector = Detector.create_test_instance()
    x = make_annotation_problem(n_timepoints=50, estimator_type="None")
    fit_detector = detector.fit(x)
    assert issubclass(detector.__class__, BaseDetector)
    assert issubclass(fit_detector.__class__, BaseDetector)
    assert isinstance(fit_detector, Detector)


@pytest.mark.parametrize("Detector", all_detectors)
def test_detector_predict(Detector):
    """Test predict method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=60)
    y = detector.fit_predict(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize("Detector", all_detectors)
def test_detector_transform(Detector):
    """Test transform method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=61)
    y = detector.fit_transform(x)
    assert isinstance(y, (pd.Series, pd.DataFrame))
    assert len(x) == len(y)


@pytest.mark.parametrize("Detector", all_detectors)
def test_detector_score_transform(Detector):
    """Test score_transform method output."""
    detector = Detector.create_test_instance()
    x = generate_anomalous_data(means=10, random_state=62)
    try:
        y = detector.fit(x).score_transform(x)
        assert isinstance(y, (pd.Series, pd.DataFrame))
    except NotImplementedError:
        pass
