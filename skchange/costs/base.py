"""Cost functions as interval evaluators."""

import numpy as np
from numpy.typing import ArrayLike

from skchange.base import BaseIntervalEvaluator
from skchange.utils.validation.intervals import check_array_intervals


class BaseCost(BaseIntervalEvaluator):
    """Base class template for cost functions."""

    def __init__(self, param=None):
        self.param = param
        super().__init__()

    def _check_param(self, param, X):
        if param is None:
            return None
        return self._check_fixed_param(param, X)

    def _check_fixed_param(self, param, X):
        return param

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 1

    def _check_intervals(self, intervals: ArrayLike) -> np.ndarray:
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=2)

    def _evaluate(self, intervals: np.ndarray) -> np.ndarray:
        starts, ends = intervals[:, 0], intervals[:, 1]
        if self.param is None:
            costs = self._evaluate_optim_param(starts, ends)
        else:
            costs = self._evaluate_fixed_param(starts, ends)

        return costs

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        raise NotImplementedError("abstract method")

    def _evaluate_fixed_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        raise NotImplementedError("abstract method")
