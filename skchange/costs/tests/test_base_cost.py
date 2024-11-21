import numpy as np
import pytest

from skchange.costs.base import BaseCost


class DummyCost(BaseCost):
    def _evaluate_optim_param(self, starts, ends):
        return np.array([[1] * len(starts)])

    def _evaluate_fixed_param(self, starts, ends):
        return np.array([[2] * len(starts)])


def test_base_cost_init():
    cost = BaseCost()
    assert cost.param is None


def test_base_cost_check_param():
    cost = BaseCost()
    X = np.array([1, 2, 3])
    assert cost._check_param(None, X) is None
    assert cost._check_param(1, X) == 1


def test_base_cost_check_fixed_param():
    cost = BaseCost()
    X = np.array([1, 2, 3])
    assert cost._check_fixed_param(1, X) == 1


def test_base_cost_evaluate():
    intervals = np.array([[0, 1], [1, 2]])
    cost = DummyCost()
    assert np.array_equal(cost._evaluate(intervals), np.array([[1, 1]]))

    cost = DummyCost(param=1)
    assert np.array_equal(cost._evaluate(intervals), np.array([[2, 2]]))


def test_base_cost_evaluate_optim_param():
    cost = BaseCost()
    with pytest.raises(NotImplementedError):
        cost._evaluate_optim_param(np.array([0]), np.array([1]))


def test_base_cost_evaluate_fixed_param():
    cost = BaseCost()
    with pytest.raises(NotImplementedError):
        cost._evaluate_fixed_param(np.array([0]), np.array([1]))
