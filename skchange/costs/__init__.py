"""Cost functions for cost-based change and anomaly detection."""

from skchange.costs.base import BaseCost
from skchange.costs.gaussian_cov_cost import GaussianCovCost
from skchange.costs.gaussian_var_cost import GaussianVarCost
from skchange.costs.l2_cost import L2Cost

BASE_COSTS = [
    BaseCost,
]
COSTS = [
    GaussianCovCost,
    GaussianVarCost,
    L2Cost,
]

__all__ = BASE_COSTS + COSTS
