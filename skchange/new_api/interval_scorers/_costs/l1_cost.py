"""L1 cost function."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_median


@njit
def l1_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Calculate the L1 cost for an optimal median location for each segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    mle_locations = np.zeros(n_columns)

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]
        mle_locations = col_median(segment, output_array=mle_locations)
        costs[i, :] = np.sum(np.abs(segment - mle_locations), axis=0)

    return costs


class L1Cost(BaseCost):
    r"""L1 (absolute error) cost function.

    Computes the sum of absolute deviations from the sample median for each segment.

    .. math::
        C(X_{s:e}) = \sum_{i=s}^{e-1} |x_i - \text{median}(X_{s:e})|

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import L1Cost
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> cost = L1Cost()
    >>> cost.fit(X)
    L1Cost()
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> cost.evaluate(cache, interval_specs)
    """

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate L1 cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            L1 costs for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return l1_cost(starts, ends, cache["X"])

    def get_default_penalty(self) -> float:
        """Get default penalty value for the fitted L1 cost."""
        penalty = bic_penalty(self.n_features_in_, self.n_samples_in_)
        # BIC works on a squared error scale, while L1 is on an absolute error scale.
        return np.sqrt(penalty)
