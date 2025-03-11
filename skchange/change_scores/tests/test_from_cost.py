import numpy as np
import pytest

from skchange.change_scores.from_cost import ChangeScore, to_change_score
from skchange.costs import ALL_COSTS, COSTS, BaseCost


@pytest.mark.parametrize("cost_class", COSTS)
def test_change_score_with_costs(cost_class):
    cost_instance = cost_class()
    change_score = ChangeScore(cost=cost_instance)
    X = np.random.randn(100, 1)
    change_score.fit(X)
    cuts = np.array([[0, 50, 100]])
    scores = change_score._evaluate(cuts)
    assert scores.shape == (1, 1)


@pytest.mark.parametrize("evaluator", ALL_COSTS)
def test_to_change_score(evaluator: type[BaseCost]):
    init_params = evaluator.get_test_params()[0]
    cost_instance = evaluator(**init_params)
    change_score = to_change_score(cost_instance)
    assert isinstance(change_score, ChangeScore)


def test_to_change_score_invalid():
    with pytest.raises(ValueError):
        to_change_score("invalid_evaluator")
