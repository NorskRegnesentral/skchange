"""Scores for change detection."""

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseChangeScore, BaseCost


class CostBasedChangeScore(BaseChangeScore):
    """Change score based on a cost function."""

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    def _build_jitted_precompute(self):
        self.jitted_precompute = self.cost.jitted_precompute

    def _build_jitted_compute(self):
        cost_jitted_compute = self.cost.jitted_compute

        @njit(cache=True)
        def cost_based_change_score(
            precomputed_params: tuple[np.ndarray, np.ndarray, np.ndarray],
            starts: np.ndarray,
            ends: np.ndarray,
            splits: np.ndarray,
        ) -> np.ndarray:
            pre_split_cost = cost_jitted_compute(precomputed_params, starts, splits)
            post_split_cost = cost_jitted_compute(precomputed_params, splits + 1, ends)
            full_cost = cost_jitted_compute(precomputed_params, starts, ends)
            return full_cost - pre_split_cost - post_split_cost

        self.jitted_compute = cost_based_change_score
