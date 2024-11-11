"""Savings for cost-based anomaly detection with known baseline parameters."""

import numpy as np
from numba import njit

from skchange.subset_evaluation.base import NumbaSubsetEvaluator
from skchange.subset_evaluation.costs import NumbaCost
from skchange.subset_evaluation.utils import subset_interval


class NumbaSaving(NumbaSubsetEvaluator):
    """Base class template for saving functions."""

    def _build_subset(self):
        self._subset = subset_interval


class CostBasedSaving(NumbaSaving):
    """Base class template for saving functions."""

    def __init__(
        self,
        baseline_cost: NumbaCost,
    ):
        if baseline_cost.param is None:
            raise ValueError(
                "The `param` of the baseline cost function must be set."
                + " It represents the baseline parameter to calculate the saving from."
            )
        self.baseline_cost = baseline_cost
        self.optimised_cost = baseline_cost.clone().set_params({"param": None})
        super().__init__()

    def _build_evaluate(self):
        baseline_cost = self.baseline_cost._evaluate
        optimised_cost = self.optimised_cost._evaluate

        @njit(cache=True)
        def _evaluate_saving(X: np.ndarray) -> float:
            return baseline_cost(X) - optimised_cost(X)

        self._evaluate = _evaluate_saving

    def _build_precompute(self):
        baseline_precompute = self.baseline_cost._precompute
        optimised_precompute = self.optimised_cost._precompute

        @njit(cache=True)
        def precompute(X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
            return baseline_precompute(X), optimised_precompute(X)

        self._precompute = precompute

    def _build_evaluate_vectorized(self):
        baseline_cost_vectorized = self.baseline_cost._evaluate_vectorized
        optimised_cost_vectorized = self.optimised_cost._evaluate_vectorized

        @njit(cache=True)
        def evaluate_vectorized(
            precomputed: tuple[list[np.ndarray], list[np.ndarray]],
            subsetter: np.ndarray,
        ) -> np.ndarray:
            precomputed_baseline, precomputed_optimised = precomputed
            cost_baseline = baseline_cost_vectorized(precomputed_baseline, subsetter)
            cost_optimised = optimised_cost_vectorized(precomputed_optimised, subsetter)
            return cost_baseline - cost_optimised

        self._evaluate_vectorized = evaluate_vectorized
