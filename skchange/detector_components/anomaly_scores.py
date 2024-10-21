"""Scores for anomaly detection."""

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseAnomalyScore, BaseCost


class CostBasedAnomalyScore(BaseAnomalyScore):
    """Anomaly score based on a cost function."""

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    def _build_jitted_precompute(self):
        self.jitted_precompute = self.cost.jitted_precompute

    def _build_jitted_compute(self):
        cost = self.cost.jitted_compute

        @njit(cache=True)
        def cost_based_anomaly_score(
            precomputed: tuple,
            starts: np.ndarray,
            ends: np.ndarray,
            anomaly_starts: np.ndarray,
            anomaly_ends: np.ndarray,
        ) -> np.ndarray:
            pre_anomaly_cost = cost(precomputed, starts, anomaly_starts - 1)
            anomaly_cost = cost(precomputed, anomaly_starts, anomaly_ends)
            post_anomaly_cost = cost(precomputed, anomaly_ends + 1, ends)
            full_cost = cost(precomputed, starts, ends)
            return full_cost - (pre_anomaly_cost + anomaly_cost + post_anomaly_cost)

        self.jitted_compute = cost_based_anomaly_score
