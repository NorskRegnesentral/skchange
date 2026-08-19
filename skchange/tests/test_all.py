"""Basic sklearn compatibility tests for estimators in ``skchange``."""

from sklearn.utils.estimator_checks import parametrize_with_checks

from skchange.detectors.tests._registry import DETECTOR_TEST_INSTANCES
from skchange.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

ALL_ESTIMATORS = [
    estimator
    for estimator in (INTERVAL_SCORER_TEST_INSTANCES + DETECTOR_TEST_INSTANCES)
    if not estimator.__sklearn_tags__().input_tags.timestamps  # Experimental
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

    failed = {
        "check_methods_subset_invariance": (
            "Subset checks may create single-sample inputs, but change detection "
            "requires at least two samples."
        ),
        "check_methods_sample_order_invariance": (
            "Change detection is for time series; sample order is semantically "
            "meaningful and not invariant under permutation."
        ),
    }

    if not tags.input_tags.multivariate:
        # sklearn's checks generate multivariate data and call fit; there is no
        # sklearn tag to signal "univariate only", so we mark all affected checks.
        # The most important of these are re-implemented for univariate data, currently
        # only relevant for the FPOP detector in skchange/detectors/tests/test_fpop.py.
        reason = (
            "Estimator only accepts univariate input (n_features=1); "
            "sklearn's checks pass multivariate data to fit."
        )
        failed.update(
            dict.fromkeys(
                (
                    "check_dict_unchanged",
                    "check_dont_overwrite_parameters",
                    "check_dtype_object",
                    "check_estimators_dtypes",
                    "check_estimators_fit_returns_self",
                    "check_estimators_nan_inf",
                    "check_estimators_overwrite_params",
                    "check_estimators_pickle",
                    "check_f_contiguous_array_estimator",
                    "check_fit2d_1sample",
                    "check_fit2d_predict1d",
                    "check_fit_check_is_fitted",
                    "check_fit_idempotent",
                    "check_fit_score_takes_y",
                    "check_n_features_in",
                    "check_n_features_in_after_fitting",
                    "check_pipeline_consistency",
                    "check_positive_only_tag_during_fit",
                    "check_readonly_memmap_input",
                ),
                reason,
            )
        )

    return failed


@parametrize_with_checks(
    ALL_ESTIMATORS,
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_compatibility(estimator, check):
    """Run sklearn's estimator checks on all skchange estimators."""
    check(estimator)
