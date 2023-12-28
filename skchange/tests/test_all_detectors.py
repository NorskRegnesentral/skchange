"""Tests for all annotators/detectors in skchange."""

import pytest
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.estimator_checks import check_estimator

from skchange.anomaly_detectors.tests.test_anomaly_detectors import anomaly_detectors
from skchange.change_detectors.tests.test_change_detectors import change_detectors

all_annotators = anomaly_detectors + change_detectors


@pytest.mark.parametrize("Estimator", all_annotators)
def test_sktime_annotator_compatibility(Estimator):
    """Check compatibility with sktime annotator interface."""
    if not run_test_for_class(Estimator):
        return None

    check_estimator(Estimator, raise_exceptions=True)
