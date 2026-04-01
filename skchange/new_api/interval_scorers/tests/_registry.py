"""Test instances for interval scorers in ``skchange.new_api.interval_scorers``."""

from skchange.new_api.interval_scorers._change_scores.cusum import CUSUM
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost
from skchange.new_api.interval_scorers._from_cost import CostChangeScore
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore

COST_TEST_INSTANCES = [
    (L2Cost, {}),
]

CHANGE_SCORE_TEST_INSTANCES = [
    (CUSUM, {}),
]
CHANGE_SCORE_TEST_INSTANCES += [
    (CostChangeScore, {"cost": cls(**params)}) for cls, params in COST_TEST_INSTANCES
]
CHANGE_SCORE_TEST_INSTANCES += [
    (PenalisedScore, {"scorer": cls(**params), "penalty": 5.0})
    for cls, params in CHANGE_SCORE_TEST_INSTANCES
]

INTERVAL_SCORER_TEST_INSTANCES = COST_TEST_INSTANCES + CHANGE_SCORE_TEST_INSTANCES
