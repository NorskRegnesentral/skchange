"""Test instances for interval scorers in ``skchange.new_api.interval_scorers``."""

from skchange.new_api.interval_scorers._change_scores.cusum import CUSUM
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost
from skchange.new_api.interval_scorers._from_cost import CostChangeScore
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore

_COST_TEST_INSTANCES = [
    L2Cost(),
]
_CHANGE_SCORE_TEST_INSTANCES = [
    CUSUM(),
] + [CostChangeScore(cost) for cost in _COST_TEST_INSTANCES]
_PENALISED_SCORE_TEST_INSTANCES = [
    PenalisedScore(scorer) for scorer in _CHANGE_SCORE_TEST_INSTANCES
]

INTERVAL_SCORER_TEST_INSTANCES = (
    _COST_TEST_INSTANCES
    + _CHANGE_SCORE_TEST_INSTANCES
    + _PENALISED_SCORE_TEST_INSTANCES
)
