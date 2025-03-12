"""Cost functions for cost-based change and anomaly detection."""

from skchange.costs.base import BaseCost
from skchange.costs.gaussian_cost import GaussianCost
from skchange.costs.l1_cost import L1Cost
from skchange.costs.l2_cost import L2Cost
from skchange.costs.laplace_cost import LaplaceCost
from skchange.costs.linear_regression_cost import LinearRegressionCost
from skchange.costs.multivariate_gaussian_cost import MultivariateGaussianCost
from skchange.costs.multivariate_t_cost import MultivariateTCost
from skchange.costs.poisson_cost import PoissonCost

BASE_COSTS = [
    BaseCost,
]
COSTS = [
    MultivariateGaussianCost,
    MultivariateTCost,
    GaussianCost,
    LaplaceCost,
    L1Cost,
    L2Cost,
]
INTEGER_COSTS = [
    PoissonCost,
]
REGRESSION_COSTS = [
    LinearRegressionCost,
]

ALL_COSTS = COSTS + INTEGER_COSTS + REGRESSION_COSTS

__all__ = BASE_COSTS + COSTS + INTEGER_COSTS
