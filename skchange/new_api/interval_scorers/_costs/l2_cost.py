"""L2 cost function."""

__author__ = ["Tveten"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def l2_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of 0-entries as the first
        row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - partial_sums**2 / n
    return costs


class L2Cost(BaseCost):
    r"""L2 (squared error) cost function.

    Computes the sum of squared deviations from the sample mean for each segment.

    .. math::
        C(X_{s:e}) = \sum_{i=s}^{e-1} ||x_i - \bar{x}_{s:e}||^2

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import L2Cost
    >>> X = np.random.randn(100, 2)
    >>> cost = L2Cost()
    >>> cost.fit(X)
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> costs = cost.evaluate(cache, interval_specs)
    """

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for efficient interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Cached cumulative sums.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate L2 cost on intervals.

        Parameters
        ----------
        cache : dict
            The output from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            L2 costs for each interval and features.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        sums, sums2 = cache["sums"], cache["sums2"]
        return l2_cost(starts, ends, sums, sums2)
