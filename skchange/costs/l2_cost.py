"""L2 cost."""

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from skchange.costs.base import BaseCost
from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.njit import njit
from skchange.utils.numba.stats import col_cumsum
from skchange.utils.validation.data import as_2d_array


@njit(cache=True)
def l2_cost_optim(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    sample_sizes: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

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
    """Calculate the L2 cost for a fixed constant mean for each segment.

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
    """L2 cost of a constant mean.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed mean for the cost calculation. If None, the optimal mean is calculated.
    """

    def __init__(self, param: Optional[float | ArrayLike] = None):
        super().__init__(param)

    def _check_fixed_param(self, param: float | ArrayLike, X: np.ndarray) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : float or array-like
            The input parameter to check.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean = np.asarray(param)
        if len(mean) != 1 and len(mean) != X.shape[1]:
            raise ValueError(
                f"param must have length 1 or X.shape[1], got {len(mean)}."
            )
        return mean

    def _fit(self, X: ArrayLike, y=None):
        """Fit the cost interval evaluator.

        This method precomputes quantities that speed up the cost evaluation.

        Parameters
        ----------
        X : array-like
            Input data.
        y: None
            Ignored. Included for API consistency by convention.
        """
        X = as_2d_array(X)
        self._mean = self._check_param(self.param, X)

        self.sums_ = col_cumsum(X, init_zero=True)
        self.sums2_ = col_cumsum(X**2, init_zero=True)
        self.sample_sizes_ = col_repeat(np.arange(0, X.shape[0] + 1), X.shape[1])

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            Costs for each interval.
        """
        return l2_cost_optim(starts, ends, self.sums_, self.sums2_, self.sample_sizes_)

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            Costs for each interval.
        """
        return l2_cost_fixed(
            starts, ends, self.sums_, self.sums2_, self.sample_sizes_, self._mean
        )
