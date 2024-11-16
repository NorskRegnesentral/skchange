"""Cost functions for cost-based change and anomaly detection."""

from skchange.costs.base import BaseCost
from skchange.costs.l2_cost import L2Cost

BASE_COSTS = [
    BaseCost,
]
COSTS = [
    L2Cost,
]

__all__ = BASE_COSTS + COSTS
