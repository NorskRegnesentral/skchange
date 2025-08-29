import numpy as np
import pytest

from skchange.costs._rank_cost import RankCost


def test_rank_cost_single_variable_no_change():
    # Single variable, no change
    X = np.arange(10).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Should be a single value, negative and not zero
    assert costs.shape == (1, 1)
    assert costs[0, 0] < 0

def test_rank_cost_single_variable_with_change():
    # Single variable, clear change in distribution
    X = np.concatenate([np.random.rand(5), np.random.rand(5) * 10]).reshape(-1, 1)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 0, 5])
    ends = np.array([10, 5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    # Both segments should have negative costs, but different values
    assert costs.shape == (3, 1)
    assert costs[0, 0] - (costs[1, 0] + costs[2, 0]) > 0

def test_rank_cost_multivariate_no_change():
    # Multivariate, no change
    X = np.tile(np.arange(10), (2, 1)).T
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0])
    ends = np.array([10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (1, 1)
    assert costs[0, 0] < 0

def test_rank_cost_multivariate_with_change():
    # Multivariate, change in one variable
    X = np.zeros((10, 2))
    X[:5, 0] = 1 * np.random.rand(5)
    X[5:, 0] = 10 * np.random.rand(5)
    X[:, 1] = np.arange(10)
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] != pytest.approx(costs[1, 0])

def test_rank_cost_multivariate_change_both_vars():
    # Multivariate, change in both variables
    X = np.zeros((10, 2))
    X[:5, 0] = 1
    X[5:, 0] = 10
    X[:5, 1] = 2
    X[5:, 1] = 20
    cost = RankCost()
    cost.fit(X)
    starts = np.array([0, 5])
    ends = np.array([5, 10])
    costs = cost._evaluate_optim_param(starts, ends)
    assert costs.shape == (2, 1)
    assert costs[0, 0] != pytest.approx(costs[1, 0])

def test_rank_cost_min_size_property():
    cost = RankCost()
    assert cost.min_size == 2

def test_rank_cost_model_size():
    cost = RankCost()
    assert cost.get_model_size(3) == 6
