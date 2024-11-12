"""Cost functions for interval evaluation."""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from skchange.interval_evaluators.base import BaseIntervalEvaluator
from skchange.interval_evaluators.utils import check_array_intervals
from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.njit import njit
from skchange.utils.numba.stats import col_cumsum
from skchange.utils.validation.data import as_2d_array


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


@njit(cache=True)
def l2_cost_optim(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    sample_sizes: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian mean likelihood cost for each segment.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    costs : `np.ndarray`
        Costs for each segment.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    cost_matrix = partial_sums2 - partial_sums**2 / sample_sizes[ends - starts]
    costs = np.sum(cost_matrix, axis=1)
    return costs


@njit(cache=True)
def l2_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    sample_sizes: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian mean likelihood cost for each segment.

    Parameters
    ----------
    mean : `np.ndarray`
        Fixed mean for the cost calculation.
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    costs : `np.ndarray`
        Costs for each segment.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    cost_matrix = (
        partial_sums2 - 2 * mean * partial_sums + sample_sizes[ends - starts] * mean**2
    )
    costs = np.sum(cost_matrix, axis=1)
    return costs


class L2Cost(BaseCost):
    """L2 cost."""

    def __init__(self, param: Optional[float | ArrayLike] = None):
        super().__init__(param)

    def _check_fixed_param(
        self, param: Optional[float | ArrayLike], X: np.ndarray
    ) -> np.ndarray:
        mean = np.asarray(param)
        if len(mean) != 1 and len(mean) != X.shape[1]:
            raise ValueError(
                f"param must have length 1 or X.shape[1], got {len(mean)}."
            )
        return mean

    def _fit(self, X: ArrayLike, y=None):
        X = as_2d_array(X)
        self._mean = self._check_param(self.param, X)

        self.sums_ = col_cumsum(X, init_zero=True)
        self.sums2_ = col_cumsum(X**2, init_zero=True)
        self.sample_sizes_ = col_repeat(np.arange(0, X.shape[0] + 1), X.shape[1])

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        return l2_cost_optim(starts, ends, self.sums_, self.sums2_, self.sample_sizes_)

    def _evaluate_fixed_param(self, starts, ends):
        return l2_cost_fixed(
            starts, ends, self.sums_, self.sums2_, self.sample_sizes_, self._mean
        )
