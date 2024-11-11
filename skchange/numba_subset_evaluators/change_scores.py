"""Scores for change detection."""

import numpy as np
from numba import njit

from skchange.numba_subset_evaluators.base import NumbaSubsetEvaluator
from skchange.numba_subset_evaluators.costs import NumbaCost
from skchange.numba_subset_evaluators.utils import split_intervals


class NumbaChangeScore(NumbaSubsetEvaluator):
    """Base class template for change scores."""

    def _build_subset(self):
        @njit(cache=True)
        def _subset(X: np.ndarray, subsetter: np.ndarray) -> list[np.ndarray]:
            if len(subsetter) != 3:
                raise ValueError(
                    "The subsetter for change scores must have three elements."
                )
            return split_intervals(X, subsetter)

        self._subset = _subset


class CostBasedChangeScore(NumbaChangeScore):
    def __init__(self, cost: NumbaCost):
        self.cost = cost
        super().__init__()

    def _build_evaluate(self):
        eval_cost = self.cost._evaluate

        @njit(cache=True)
        def _evaluate(X_subsets: list[np.ndarray]) -> float:
            cost_overall = eval_cost(X_subsets[0])
            cost_1 = eval_cost(X_subsets[1])
            cost_2 = eval_cost(X_subsets[2])
            return cost_overall - (cost_1 + cost_2)

        self._evaluate = _evaluate

    def _build_precompute(self):
        self._precompute = self.cost._precompute

    def _build_evaluate_vectorized(self):
        eval_cost_vectorized = self.cost._evaluate_vectorized

        @njit(cache=True)
        def evaluate_vectorized(
            precomputed: np.ndarray, subsetter: np.ndarray
        ) -> np.ndarray:
            if subsetter.shape[1] != 3:
                shape = subsetter.shape
                raise ValueError(
                    f"The subsetter must have three columns. Got shape {shape}."
                )
            cost_overall = eval_cost_vectorized(precomputed, subsetter[:, [0, 2]])
            cost_1 = eval_cost_vectorized(precomputed, subsetter[:, [0, 1]])
            cost_2 = eval_cost_vectorized(precomputed, subsetter[:, [1, 2]])
            return cost_overall - (cost_1 + cost_2)

        self._evaluate_vectorized = evaluate_vectorized
