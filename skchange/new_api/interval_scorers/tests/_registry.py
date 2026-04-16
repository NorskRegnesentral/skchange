"""Test instances for interval scorers in ``skchange.new_api.interval_scorers``."""

from skchange.new_api.interval_scorers import (
    CUSUM,
    CostChangeScore,
    EDFCost,
    GaussianCost,
    GaussianSaving,
    L1Cost,
    L1Saving,
    L2Cost,
    L2Saving,
    PenalisedScore,
)

# ---------------------------------------------------------------------------
# Raw instances of each type of interval scorer
# ---------------------------------------------------------------------------
_COSTS = [
    EDFCost(),
    GaussianCost(),
    L1Cost(),
    L2Cost(),
]
_CHANGE_SCORES = [
    CUSUM(),
]
_SAVINGS = [
    GaussianSaving(),
    L1Saving(),
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
