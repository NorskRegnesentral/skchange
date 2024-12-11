"""Cost functions for cost-based change and anomaly detection."""

from skchange.costs.base import BaseCost
from skchange.costs.gaussian_var_cost import GaussianCost
from skchange.costs.l2_cost import L2Cost
from skchange.costs.multivariate_gaussian_cost import MultivariateGaussianCost

BASE_COSTS = [
    BaseCost,
]
COSTS = [
    MultivariateGaussianCost,
    GaussianCost,
    L2Cost,
]

__all__ = BASE_COSTS + COSTS
