import numpy as np
import pytest

from skchange.costs import COSTS


def find_fixed_param_combination(cost_class):
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


def test_find_fixed_param_combination_value_error():
    class MockCost:
        @staticmethod
        def get_test_params():
            return [{"param": None}, {"param": None}]

    with pytest.raises(ValueError):
        find_fixed_param_combination(MockCost)


@pytest.mark.parametrize("CostClass", COSTS)
def test_cost_evaluation(CostClass):
    optim_cost = CostClass()
    fixed_params = find_fixed_param_combination(CostClass)
    fixed_cost = CostClass().set_params(**fixed_params)
    X = np.random.normal(size=10)
    optim_cost.fit(X)
    fixed_cost.fit(X)
    intervals = np.array([[0, 5], [5, 10]])
    optim_scores = optim_cost.evaluate(intervals)
    fixed_scores = fixed_cost.evaluate(intervals)
    assert np.all(optim_scores <= fixed_scores)
