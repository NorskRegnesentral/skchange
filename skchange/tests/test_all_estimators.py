"""Tests for all skchange estimators, both detectors and interval scorers."""

from inspect import _empty, signature

import pytest
from skbase.base import BaseObject
from sktime.base import BaseEstimator
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks

from skchange.anomaly_detectors import ANOMALY_DETECTORS
from skchange.anomaly_scores import ANOMALY_SCORES
from skchange.change_detectors import CHANGE_DETECTORS
from skchange.change_scores import CHANGE_SCORES
from skchange.compose.penalised_score import PenalisedScore
from skchange.costs import COSTS

DETECTORS = ANOMALY_DETECTORS + CHANGE_DETECTORS
INTERVAL_EVALUATORS = COSTS + CHANGE_SCORES + ANOMALY_SCORES + [PenalisedScore]
ESTIMATORS = DETECTORS + INTERVAL_EVALUATORS


@parametrize_with_checks(ESTIMATORS)
def test_sktime_compatible_estimators(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)


@pytest.mark.parametrize("Estimator", ESTIMATORS)
def test_detector_no_mutable_defaults(Estimator: BaseEstimator):
    """Ensure no detectors have mutable default arguments."""

    detector = Estimator.create_test_instance()
    sig = signature(detector.__init__)
    mutable_types = (
        list,
        dict,
        set,
        BaseEstimator,
        BaseObject,
    )
    for param in sig.parameters.values():
        if param.default is not _empty and isinstance(param.default, mutable_types):
            raise AssertionError(
                f"Mutable default argument found in {Estimator.__name__}: {param.name}"
            )
