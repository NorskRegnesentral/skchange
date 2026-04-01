"""Basic sklearn compatibility tests for estimators in ``skchange.new_api``."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from skchange.new_api.detectors.tests._registry import DETECTOR_TEST_INSTANCES
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

ALL_ESTIMATORS = [
    cls(**params)
    for cls, params in INTERVAL_SCORER_TEST_INSTANCES + DETECTOR_TEST_INSTANCES
]


def _expected_failed_checks(estimator):
    """Return sklearn checks expected to fail for known API differences.

    Notes
    -----
    skchange change detectors intentionally differ from sklearn's typical estimator
    assumptions in a few places:

    - Time-series estimators are order-sensitive by design.
    - Detectors require at least two samples.
    """
    tags = estimator.__sklearn_tags__()
    is_change_detector = tags.change_detector_tags is not None
    if not is_change_detector:
        return {}

    return {
        "check_methods_subset_invariance": (
            "Subset checks may create single-sample inputs, but change detection "
            "requires at least two samples."
        ),
        "check_methods_sample_order_invariance": (
            "Change detection is for time series; sample order is semantically "
            "meaningful and not invariant under permutation."
        ),
    }


@parametrize_with_checks(
    ALL_ESTIMATORS,
    expected_failed_checks=_expected_failed_checks,
)
def test_new_api_estimators_sklearn_compatibility(estimator, check):
    """Run sklearn's estimator checks on all skchange estimators."""
    check(estimator)
