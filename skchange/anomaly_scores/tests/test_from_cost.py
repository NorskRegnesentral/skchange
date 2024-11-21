import numpy as np
import pytest

from skchange.anomaly_scores.from_cost import Saving, to_saving
from skchange.costs import COSTS


def find_fixed_param_combinations(cost_class):
    """Find the first fixed parameter combination in the test parameters of a cost."""
    test_param_sets = cost_class.get_test_params()
    fixed_test_param_set = None
    for param_set in test_param_sets:
        if "param" in param_set and param_set["param"] is not None:
            fixed_test_param_set = param_set
            break

    if fixed_test_param_set is None:
        raise ValueError(
            f"No fixed `param` argument found in `get_test_params()` of"
            f" the cost class {cost_class.__name__}"
        )

    return fixed_test_param_set


def test_find_fixed_param_combinations_value_error():
    class MockCost:
        @staticmethod
        def get_test_params():
            return [{"param": None}, {"param": None}]

    with pytest.raises(ValueError):
        find_fixed_param_combinations(MockCost)


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_init(cost_class):
    param = find_fixed_param_combinations(cost_class)
    baseline_cost = cost_class().set_params(**param)

    saving = Saving(baseline_cost)
    assert saving.baseline_cost == baseline_cost
    assert saving.optimised_cost.param is None


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_min_size(cost_class):
    param = find_fixed_param_combinations(cost_class)
    cost_instance = cost_class().set_params(**param)

    saving = Saving(baseline_cost=cost_instance)
    assert saving.min_size == cost_instance.min_size


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_fit(cost_class):
    param = find_fixed_param_combinations(cost_class)
    cost_instance = cost_class().set_params(**param)

    saving = Saving(baseline_cost=cost_instance)
    X = np.random.randn(100, 1)
    saving.fit(X)
    assert saving.baseline_cost.is_fitted
    assert saving.optimised_cost.is_fitted


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_evaluate(cost_class):
    param = find_fixed_param_combinations(cost_class)
    cost_instance = cost_class().set_params(**param)

    saving = Saving(baseline_cost=cost_instance)
    X = np.random.randn(100, 1)
    saving.fit(X)
    intervals = np.array([[0, 10], [10, 20], [20, 30]])
    savings = saving.evaluate(intervals)
    assert savings.shape == (3, 1)


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_init_error(cost_class):
    with pytest.raises(
        ValueError, match="The baseline cost must have a fixed parameter."
    ):
        Saving(baseline_cost=cost_class().set_params(param=None))


def test_to_saving_error():
    with pytest.raises(
        ValueError, match="evaluator must be an instance of BaseSaving or BaseCost."
    ):
        to_saving("invalid_evaluator")
