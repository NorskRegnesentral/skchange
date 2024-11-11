"""Scores for anomaly detection."""

import numpy as np
from numba import njit

from skchange.numba_subset_evaluators.base import NumbaSubsetEvaluator
from skchange.numba_subset_evaluators.change_scores import (
    CostBasedChangeScore,
    NumbaChangeScore,
)
from skchange.numba_subset_evaluators.costs import NumbaCost
from skchange.numba_subset_evaluators.utils import split_intervals


class NumbaAnomalyScore(NumbaSubsetEvaluator):
    """Base class template for anomaly scores."""

    def _build_subset(self):
        @njit(cache=True)
        def _subset(X: np.ndarray, subsetter: np.ndarray) -> list[np.ndarray]:
            if len(subsetter) != 4:
                raise ValueError(
                    "The subsetter for anomaly scores must have four elements."
                )
            X_subsets = split_intervals(X, subsetter)
            X_combined = X_subsets[0]
            X_normal = np.concatenate((X_subsets[1], X_subsets[3]))
            X_anomaly = X_subsets[2]
            return X_combined, X_normal, X_anomaly

        self._subset = _subset


class CostBasedAnomalyScore(NumbaAnomalyScore):
    """Anomaly score based on a cost function."""

    def __init__(self, cost: NumbaCost):
        self.cost = cost
        super().__init__()

    def _build_evaluate(self):
        cost = self.cost._evaluate

        @njit(cache=True)
        def _evaluate(X_subsets: list[np.ndarray]) -> float:
            X_combined = X_subsets[0]
            X_normal = X_subsets[1]
            X_anomaly = X_subsets[2]
            return cost(X_combined) - (cost(X_normal) + cost(X_anomaly))

        self._evaluate = _evaluate


class CostBasedAnomalyScore(NumbaAnomalyScore):
    def __init__(self, evaluator: NumbaCost | NumbaChangeScore):
        self.evaluator = evaluator
        if isinstance(evaluator, NumbaCost):
            self.change_score = CostBasedChangeScore(evaluator)
        elif isinstance(evaluator, NumbaChangeScore):
            self.change_score = evaluator
        else:
            raise ValueError(
                "The evaluator must be either a `NumbaCost` or a `NumbaChangeScore`."
            )
        super().__init__()

    def _build_evaluate(self):
        self._evaluate = self.change_score._evaluate
