"""Test instances for interval scorers in ``skchange.new_api.interval_scorers``."""

from skchange.new_api.interval_scorers import (
    CUSUM,
    ContinuousLinearTrendScore,
    CostChangeScore,
    EDFCost,
    ESACScore,
    GaussianCost,
    GaussianSaving,
    L1Cost,
    L1Saving,
    L2Cost,
    L2Saving,
    LaplaceCost,
    LaplaceSaving,
    LinearRegressionCost,
    LinearRegressionSaving,
    LinearTrendCost,
    LinearTrendSaving,
    MultivariateGaussianCost,
    MultivariateGaussianSaving,
    MultivariateGaussianScore,
    MultivariateTCost,
    MultivariateTSaving,
    PenalisedScore,
    PoissonCost,
    PoissonSaving,
    RankCost,
    RankScore,
    is_penalised_score,
)

# ---------------------------------------------------------------------------
# Raw instances of each type of interval scorer
# ---------------------------------------------------------------------------
_COSTS = [
    EDFCost(),
    GaussianCost(),
    L1Cost(),
    L2Cost(),
    LaplaceCost(),
    LinearRegressionCost(),
    LinearTrendCost(),
    MultivariateGaussianCost(),
    MultivariateTCost(),
    PoissonCost(),
    RankCost(),
]
_CHANGE_SCORES = [
    CUSUM(),
    ContinuousLinearTrendScore(),
    ESACScore(),
    MultivariateGaussianScore(),
    RankScore(),
]
_SAVINGS = [
    GaussianSaving(),
    L1Saving(),
    L2Saving(),
    LaplaceSaving(),
    LinearRegressionSaving(),
    LinearTrendSaving(),
    MultivariateGaussianSaving(),
    MultivariateTSaving(),
    PoissonSaving(),
]

# ---------------------------------------------------------------------------
# Composite instances
# ---------------------------------------------------------------------------
_COST_COMPOSITES = [CostChangeScore(cost) for cost in _COSTS]
_PENALISED_SCORES = [
    PenalisedScore(scorer)
    for scorer in _CHANGE_SCORES + _SAVINGS + _COST_COMPOSITES
    if not is_penalised_score(scorer)
]

# ---------------------------------------------------------------------------
# All test instances
# ---------------------------------------------------------------------------
INTERVAL_SCORER_TEST_INSTANCES = (
    _COSTS + _CHANGE_SCORES + _SAVINGS + _COST_COMPOSITES + _PENALISED_SCORES
)
