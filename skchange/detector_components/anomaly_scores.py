"""Scores for anomaly detection."""

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseAnomalyScore, BaseCost
from skchange.detector_components.utils import identity_func


class CostBasedAnomalyScore(BaseAnomalyScore):
    """Anomaly score based on a cost function."""

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    def _build_jitted_precompute(self):
        self.jitted_precompute = identity_func

    def _build_jitted_compute(self):
        generic_cost = self.cost.jitted_compute_generic

        @njit(cache=True)
        def cost_based_anomaly_score(
            X: np.ndarray,
            starts: np.ndarray,
            ends: np.ndarray,
            anomaly_starts: np.ndarray,
            anomaly_ends: np.ndarray,
        ) -> np.ndarray:
            scores = np.zeros(len(starts), dtype=np.float64)
            for i, start, end, anomaly_start, anomaly_end in zip(
                range(len(starts)), starts, ends, anomaly_starts, anomaly_ends
            ):
                anomaly_cost = generic_cost(X[anomaly_start : anomaly_end + 1])
                baseline_data = np.concatenate(
                    (X[start:anomaly_start], X[anomaly_end + 1 : end + 1])
                )
                baseline_cost = generic_cost(baseline_data)
                full_cost = generic_cost(X[start : end + 1])
                scores[i] = full_cost - (baseline_cost + anomaly_cost)

            return scores

        self.jitted_compute = cost_based_anomaly_score
