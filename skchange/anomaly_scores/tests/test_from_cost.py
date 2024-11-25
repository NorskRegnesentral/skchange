import numpy as np
import pytest

from skchange.anomaly_scores import (
    LocalAnomalyScore,
    Saving,
    to_local_anomaly_score,
    to_saving,
)
from skchange.costs import COSTS
from skchange.costs.tests.test_all_costs import find_fixed_param_combination


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_init(cost_class):
    param = find_fixed_param_combination(cost_class)
    baseline_cost = cost_class().set_params(**param)

    saving = Saving(baseline_cost)
    assert saving.baseline_cost == baseline_cost
    assert saving.optimised_cost.param is None


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_min_size(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)
    saving = Saving(baseline_cost=cost)

    np.random.seed(132)
    X = np.random.randn(100, 1)
    cost.fit(X)
    saving.fit(X)
    assert saving.min_size == cost.min_size


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_fit(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)

    saving = Saving(baseline_cost=cost)
    X = np.random.randn(100, 1)
    saving.fit(X)
    assert saving.baseline_cost.is_fitted
    assert saving.optimised_cost.is_fitted


@pytest.mark.parametrize("cost_class", COSTS)
def test_saving_evaluate(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)

    saving = Saving(baseline_cost=cost)
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


@pytest.mark.parametrize("cost_class", COSTS)
def test_to_local_anomaly_score_with_base_cost(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)
    local_anomaly_score = to_local_anomaly_score(cost)
    assert isinstance(local_anomaly_score, LocalAnomalyScore)
    assert local_anomaly_score.cost == cost


@pytest.mark.parametrize("cost_class", COSTS)
def test_to_local_anomaly_score_with_local_anomaly_score(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)
    local_anomaly_score_instance = LocalAnomalyScore(cost=cost)
    result = to_local_anomaly_score(local_anomaly_score_instance)
    assert result is local_anomaly_score_instance


@pytest.mark.parametrize("cost_class", COSTS)
def test_local_anomaly_score_evaluate(cost_class):
    param = find_fixed_param_combination(cost_class)
    cost = cost_class().set_params(**param)
    local_anomaly_score = LocalAnomalyScore(cost=cost)
    X = np.random.randn(100, 1)
    local_anomaly_score.fit(X)
    cuts = np.array([[0, 5, 10, 15], [5, 10, 15, 20], [10, 15, 20, 25]])
    scores = local_anomaly_score.evaluate(cuts)
    assert scores.shape == (3, 1)


def test_to_local_anomaly_score_error():
    with pytest.raises(
        ValueError,
        match="evaluator must be an instance of BaseLocalAnomalyScore or BaseCost.",
    ):
        to_local_anomaly_score("invalid_evaluator")
