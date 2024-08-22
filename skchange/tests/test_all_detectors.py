"""Tests for all annotators/detectors in skchange."""

from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from skchange.anomaly_detectors.tests.test_anomaly_detectors import anomaly_detectors
from skchange.change_detectors.tests.test_change_detectors import change_detectors

all_detectors = anomaly_detectors + change_detectors


@parametrize_with_checks(all_detectors)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
