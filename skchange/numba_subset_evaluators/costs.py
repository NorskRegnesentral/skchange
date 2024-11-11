"""Cost functions."""

__author__ = ["Tveten"]

from typing import Optional

import numpy as np
from numba import njit

from skchange.numba_subset_evaluators.base import (
    NumbaSubsetEvaluator,
    build_default_evaluate_vectorized,
)
from skchange.numba_subset_evaluators.utils import identity, subset_interval
from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


class NumbaCost(NumbaSubsetEvaluator):
    """Base class template for cost functions."""

    def __init__(
        self,
        param: Optional[float | np.ndarray | list[float] | list[np.ndarray]] = None,
        min_size: int = 1,
    ):
        self.param = param
        self.min_size = min_size
        super().__init__()

    def _build_subset(self):
        self._subset = subset_interval

    def _build_evaluate_optim(self):
        # def evaluate_optimised_cost(X: np.ndarray) -> float:
        #     <code>
        #     return value
        #
        # self._evaluate = evaluate_optimised_cost
        raise NotImplementedError("abstract method")

    def _build_evaluate_fixed(self):
        # def evaluate_fixed_cost(X: np.ndarray) -> float:
        #     <code>
        #     return value
        #
        # self._evaluate= evaluate_fixed_cost
        raise NotImplementedError("abstract method")

    def _build_evaluate(self):
        if self.param is None:
            self._build_evaluate_optim()
        else:
            self._build_evaluate_fixed()

    def _build_precompute_vectorized_optim(self):
        self._precompute_vectorized = identity

    def _build_precompute_vectorized_fixed(self):
        self._precompute_vectorized = identity

    def _build_precompute_vectorized(self):
        if self.param is None:
            self._build_precompute_vectorized_optim()
        else:
            self._build_precompute_vectorized_fixed()

    def _build_evaluate_vectorized_optim(self):
        self._evaluate_vectorized = build_default_evaluate_vectorized(
            self._subset, self._evaluate
        )

    def _build_evaluate_vectorized_fixed(self):
        self._evaluate_vectorized = build_default_evaluate_vectorized(
            self._subset, self._evaluate
        )

    def _build_evaluate_vectorized(self):
        if self.param is None:
            self._build_evaluate_vectorized_optim()
        else:
            self._build_evaluate_vectorized_fixed()


class L2Cost(NumbaCost):
    """L2 cost function."""

    def __init__(
        self,
        param: Optional[float | np.ndarray] = None,
        min_size: int = 1,
    ):
        super().__init__(param, min_size)

    def _build_evaluate_optim(self):
        @njit(cache=True)
        def evaluate_l2(X: np.ndarray) -> float:
            n = X.shape[0]
            sum2 = np.sum(X**2, axis=0)
            sum = np.sum(X, axis=0)
            univar_costs = sum2 - sum**2 / n
            cost = np.sum(univar_costs)
            return cost

        self._evaluate = evaluate_l2

    def _build_evaluate_fixed(self):
        mean = self.param

        @njit(cache=True)
        def evaluate_l2_fixed(X: np.ndarray) -> float:
            univar_costs = np.sum((X - mean) ** 2, axis=0)
            cost = np.sum(univar_costs)
            return cost

        self._evaluate = evaluate_l2_fixed

    def _build_precompute_vectorized_optim(self):
        @njit(cache=True)
        def precompute_l2(X: np.ndarray) -> list[np.ndarray]:
            sums = col_cumsum(X, init_zero=True)
            sums2 = col_cumsum(X**2, init_zero=True)
            sample_sizes = col_repeat(np.arange(0, X.shape[0] + 1), X.shape[1])
            return [sums, sums2, sample_sizes]

        self._precompute_vectorized = precompute_l2

    def _build_evaluate_vectorized_optim(self):
        @njit(cache=True)
        def evaluate_l2(
            precomputed: list[np.ndarray],
            subsetter: np.ndarray,
        ) -> np.ndarray:
            sums = precomputed[0]
            sums2 = precomputed[1]
            sample_sizes = precomputed[2]
            starts = subsetter[:, 0]  # Inclusive
            ends = subsetter[:, 1]  # Exclusive
            partial_sums = sums[ends] - sums[starts]
            partial_sums2 = sums2[ends] - sums2[starts]
            cost_matrix = partial_sums2 - partial_sums**2 / sample_sizes[ends - starts]
            costs = np.sum(cost_matrix, axis=1)
            return costs

        self._evaluate_vectorized = evaluate_l2
