"""Cost functions for cost-based change and anomaly detection."""

from ._continuous_linear_trend_cost import ContinuousLinearTrendCost
from ._gaussian_cost import GaussianCost
from ._l1_cost import L1Cost
from ._l2_cost import L2Cost
from ._laplace_cost import LaplaceCost
from ._linear_regression_cost import LinearRegressionCost
from ._linear_trend_cost import LinearTrendCost
from ._multivariate_gaussian_cost import MultivariateGaussianCost
from ._multivariate_t_cost import MultivariateTCost
from ._poisson_cost import PoissonCost

COSTS = [
    ContinuousLinearTrendCost,
    MultivariateGaussianCost,
    MultivariateTCost,
    GaussianCost,
    LaplaceCost,
    L1Cost,
    L2Cost,
    LinearTrendCost,
]
INTEGER_COSTS = [
    PoissonCost,
]
REGRESSION_COSTS = [
    LinearRegressionCost,
]

ALL_COSTS = COSTS + INTEGER_COSTS + REGRESSION_COSTS

__all__ = ALL_COSTS
