"""Test instances for interval scorers in ``skchange.new_api.interval_scorers``."""

from skchange.new_api.interval_scorers._change_scores.cusum import CUSUM
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost
from skchange.new_api.interval_scorers._from_cost import CostChangeScore
from skchange.new_api.interval_scorers._penalised_score import PenalisedScore
from skchange.new_api.interval_scorers._savings.l2_saving import L2Saving

# ---------------------------------------------------------------------------
# Raw instances of each type of interval scorer
# ---------------------------------------------------------------------------
_COSTS = [
    L2Cost(),
]
_CHANGE_SCORES = [
    CUSUM(),
]
_SAVINGS = [
    L2Saving(),
]

# ---------------------------------------------------------------------------
# Composite instances
# ---------------------------------------------------------------------------
_COST_COMPOSITES = [CostChangeScore(cost) for cost in _COSTS]
_PENALISED_SCORES = [
    PenalisedScore(scorer) for scorer in _CHANGE_SCORES + _SAVINGS + _COST_COMPOSITES
]

# ---------------------------------------------------------------------------
# All test instances
# ---------------------------------------------------------------------------
INTERVAL_SCORER_TEST_INSTANCES = (
    _COSTS + _CHANGE_SCORES + _SAVINGS + _COST_COMPOSITES + _PENALISED_SCORES
)
