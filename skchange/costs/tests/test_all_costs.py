import numpy as np
import pytest

from skchange.costs import COSTS
from skchange.costs.base import BaseCost
from skchange.datasets import generate_alternating_data


def find_fixed_param_combination(cost_class: type[BaseCost]):
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
def test_l2_cost_init(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    assert cost.param is None


@pytest.mark.parametrize("CostClass", COSTS)
def test_expected_cut_entries(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    assert cost.get_required_cut_size() == 2


@pytest.mark.parametrize("CostClass", COSTS)
def test_cost_evaluation_optim_gt_fixed(CostClass: type[BaseCost]):
    if not CostClass.supports_fixed_params:
        pytest.skip(f"{CostClass.__name__} does not support fixed parameters.")

    optim_cost = CostClass()
    fixed_params = find_fixed_param_combination(CostClass)
    fixed_cost = CostClass().set_params(**fixed_params)
    np.random.seed(1001)
    X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 1]], size=20)
    optim_cost.fit(X)
    fixed_cost.fit(X)
    intervals = np.array([[0, 5], [5, 10], [10, 15], [15, 20]])
    optim_costs = optim_cost.evaluate(intervals)
    fixed_costs = fixed_cost.evaluate(intervals)
    assert np.all(optim_costs <= fixed_costs)


@pytest.mark.parametrize("CostClass", COSTS)
def test_cost_evaluation_positive(CostClass: type[BaseCost]):
    cost = CostClass.create_test_instance()
    n = 50
    df = generate_alternating_data(n_segments=1, segment_length=n, p=1, random_state=5)
    cost.fit(df)
    starts = np.arange(n - 10)
    ends = np.repeat(n - 1, len(starts))
    intervals = np.column_stack((starts, ends))
    costs = cost.evaluate(intervals)
    assert np.all(costs >= 0.0)
