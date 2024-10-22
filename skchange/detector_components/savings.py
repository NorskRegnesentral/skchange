"""Savings for cost-based anomaly detection with known baseline parameters."""

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseCost, BaseSaving
from skchange.detector_components.utils import identity_func


class CostBasedSaving(BaseSaving):
    """Anomaly score based on a cost function."""

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    def _build_jitted_precompute(self):
        self.jitted_precompute = identity_func

    def _build_jitted_compute(self):
        generic_cost = self.cost.jitted_compute_generic
        fixed_cost = self.cost.jitted_compute_fixed

        @njit(cache=True)
        def cost_based_saving(
            X: np.ndarray | tuple,
            starts: np.ndarray,
            ends: np.ndarray,
        ) -> np.ndarray:
            savings = np.zeros(len(starts), dtype=np.float64)
            for i, start, end in zip(range(len(starts)), starts, ends):
                baseline_cost = fixed_cost(X[start : end + 1])
                anomaly_cost = generic_cost(X[start : end + 1])
                savings[i] = baseline_cost - anomaly_cost

            return savings

        self.jitted_compute = cost_based_saving
