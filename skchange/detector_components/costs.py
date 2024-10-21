"""Cost functions.

What functions would be useful in cost functions.

- Compute cost over a segment (start, end).
- Compute joint cost over several disjoint segments. Useful for anomaly_score for
  example, where the cost need to be calculated for disjoint segments.
- Compute cost for any input data vector. Will work for any, also unforeseen,
  applications.
  No need/not possible to do precomputations of the type used for segments.
- Compute the cost for both fixed and estimated parameters. For example in L2, both
  for a fixed mean and for the maximum likelihood mean. Should this be two separate
  cost classes, or combined in one by using a parameter in the class constructor?


Want in addition:

- Option to write specialised detector components for common costs/scores,
  like mean, variance, etc. E.g. through precomputing quantities or through simplifying
  score expressions that involve several independent calls to the cost.
- Upon implementation of a new cost, it can be used to generate all the other types of
  components: change_scores, anomaly_scores, savings.
- Savings: Want to specify the cost and the baseline parameter outside the detector.
    Options:

    * Specify two separate cost functions. Need to check that one cost is a fixed
    version of the other somehow. Maybe this is the cleanest solution?
    * Specify the cost function, and the cost function contains both a fixed and
    maximum likelihood/estimate version.

        - Specify the baseline parameter upon construction, through init. A bit clunky
        to set the parameter for one of the functions.
        - Expose a parameter argument in the jitted fixed function. Will get messy
        for different costs.
"""

__author__ = ["Tveten"]

import numpy as np
from numba import njit

from skchange.detector_components.base import BaseCost, BaseCostOld
from skchange.detector_components.utils import init_sample_sizes, init_sums, init_sums2


class L2Cost(BaseCost):
    """L2 cost function.

    Parameters
    ----------
    fixed_mean : `float` or `np.ndarray`, default=0.0
        The default value of `mean` in `compute_fixed`. If a float, the same mean is
        used for all dimensions. If an array, the mean should have the same number of
        dimensions as the input data.
    """

    def __init__(self, fixed_mean: float | np.ndarray = 0.0):
        self.fixed_mean = fixed_mean
        self._fixed_mean = np.asarray(self.fixed_mean).reshape(-1)

    def _build_jitted_precompute(self):
        # Could be a default implementation in BaseCost that returns the input.
        @njit(cache=True)
        def init_l2_cost(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return init_sums(X), init_sums2(X), init_sample_sizes(X)

        self.jitted_precompute = init_l2_cost

    def _build_jitted_compute(self):
        # Very fast when applicable.
        # Could be a default implementation in BaseCost that uses the generic compute.
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
            cost_matrix = partial_sums2 - partial_sums**2 / weights
            costs = np.sum(cost_matrix, axis=1)
            return costs

        self.jitted_compute = l2_cost

    def _build_jitted_compute_generic(self):
        # Slower, but useful for flexibly building other components on top.
        @njit(cache=True)
        def l2_cost(x: np.ndarray) -> float:
            n = x.shape[0]
            sum2 = np.sum(x**2, axis=0)
            sum = np.sum(x, axis=0)
            univar_costs = sum2 - sum**2 / n
            cost = np.sum(univar_costs)
            return cost

        self.jitted_compute_generic = l2_cost

    def _build_jitted_compute_fixed(self):
        # Needed for building savings automatically from costs.
        mean = self._fixed_mean

        @njit(cache=True)
        def fixed_l2_cost(x: np.ndarray, mean: float | np.ndarray = mean) -> np.ndarray:
            univar_costs = np.sum((x - mean) ** 2, axis=0)
            cost = np.sum(univar_costs)
            return cost

        self.jitted_compute_fixed = fixed_l2_cost


class L2CostOld(BaseCostOld):
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


class EstimatedL2Cost(BaseCostOld):
    """L2 cost function."""

    def _build_jitted_compute(self):
        @njit(cache=True)
        def l2_cost(x: np.ndarray) -> np.ndarray:
            n = x.shape[0]
            cost_matrix = n * np.var(x, axis=0)
            cost = np.sum(cost_matrix, axis=1)
            return cost

        self.jitted_compute = l2_cost


class FixedL2Cost(BaseCostOld):
    """L2 cost for a fixed mean."""

    def __init__(self, mean: float | np.ndarray):
        self.mean = mean
        self._mean = np.asarray(mean).reshape(-1)

    def _build_jitted_compute(self):
        mean = self._mean

        @njit(cache=True)
        def fixed_l2_cost(x: np.ndarray) -> np.ndarray:
            cost_matrix = np.sum((x - mean) ** 2, axis=0)
            cost = np.sum(cost_matrix, axis=1)
            return cost

        self.jitted_compute = fixed_l2_cost
