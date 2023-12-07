import pytest

# from sktime.registry import all_estimators
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.estimator_checks import check_estimator

from skchange.change_detectors.tests.test_change_detectors import change_detectors

# todo:
# ALL_ANNOTATORS = all_estimators(
#     estimator_types="series-annotator", return_names=False
# )
all_annotators = change_detectors


@pytest.mark.parametrize("Estimator", all_annotators)
def check_sktime_annotator_compatibility(Estimator):
    """Test annotator output type."""
    if not run_test_for_class(Estimator):
        return None

    check_estimator(Estimator, raise_exceptions=True)
