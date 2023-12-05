import pytest

# from sktime.registry import all_estimators
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.estimator_checks import check_estimator

from skchange.change_detectors.pelt import Pelt

# ALL_ANNOTATORS = all_estimators(
#     estimator_types="series-annotator", return_names=False
# )
ALL_ANNOTATORS = [Pelt]


@pytest.mark.parametrize("Estimator", ALL_ANNOTATORS)
def test_output_type(Estimator):
    """Test annotator output type."""
    if not run_test_for_class(Estimator):
        return None

    check_estimator(Estimator, raise_exceptions=True)
