"""Cost functions for cost-based change and anomaly detection."""

from skchange.costs.base import BaseCost
from skchange.costs.gaussian_cost import GaussianCost
from skchange.costs.l1_cost import L1Cost, LaplaceCost
from skchange.costs.l2_cost import L2Cost
from skchange.costs.multivariate_gaussian_cost import MultivariateGaussianCost
from skchange.costs.multivariate_t_cost import MultivariateTCost

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

__all__ = BASE_COSTS + COSTS
