"""Cost functions."""

__author__ = ["Tveten"]

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseCost
from skchange.detector_components.utils import init_sample_sizes, init_sums, init_sums2


class L2Cost(BaseCost):
    """L2 cost function."""

    def _build_jitted_precompute(self):
        @njit(cache=True)
        def init_l2_cost(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return init_sums(X), init_sums2(X), init_sample_sizes(X)

        self.jitted_precompute = init_l2_cost

    def _build_jitted_compute(self):
        @njit(cache=True)
        def l2_cost(
            precomputed: tuple[np.ndarray, np.ndarray, np.ndarray],
            starts: np.ndarray,
            ends: np.ndarray,
        ) -> np.ndarray:
            sums, sums2, weights = precomputed
            partial_sums = sums[ends + 1] - sums[starts]
            partial_sums2 = sums2[ends + 1] - sums2[starts]
            weights = weights[ends - starts + 1]
            costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
            return costs

        self.jitted_compute = l2_cost
