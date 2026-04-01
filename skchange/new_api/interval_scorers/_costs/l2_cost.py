"""L2 cost function."""

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, validate_data

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.typing import ArrayLike, Self
from skchange.penalties import make_bic_penalty
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def l2_cost_optim(
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


@njit
def l2_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for a fixed constant mean for each segment.

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
    mean : np.ndarray
        Fixed mean for the cost calculation.

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
    costs = partial_sums2 - 2 * mean * partial_sums + n * mean**2
    return costs


class L2Cost(BaseCost):
    r"""L2 (squared error) cost function.

    Computes sum of squared deviations from a mean parameter.

    .. math::
        C(X) = \sum_{i=1}^{n} ||x_i - \text{mean}||^2

    Parameters
    ----------
    mean : array-like of shape (n_features,), float, or None, default=None
        Fixed mean parameter. If float, the value is broadcast across all features.
        If None, estimated as sample mean.

    Attributes
    ----------
    mean_ : ndarray of shape (n_features,)
        Fitted mean parameter.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.scorers import L2Cost
    >>>
    >>> X = np.random.randn(100, 2)
    >>>
    >>> # Estimated mode - learns mean from data
    >>> cost = L2Cost()
    >>> cost.fit(X)
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> costs = cost.evaluate(cache, interval_specs)
    >>>
    >>> # Fixed mode - user provides mean
    >>> cost_fixed = L2Cost(mean=np.array([0.0, 0.0]))
    >>> cost_fixed.fit(X)
    >>> cache = cost_fixed.precompute(X)

    Notes
    -----
    This is a simple cost function useful for detecting mean shifts in
    Gaussian data. The estimated version computes the sample mean, while
    the fixed version uses a user-specified reference mean.
    """

    def __init__(self, mean: ArrayLike | float | None = None):
        self.mean = mean

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit L2 cost to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data to learn mean from.
        y : None
            Ignored.

        Returns
        -------
        self : L2Cost
            Fitted cost function.
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)  # Sets n_features_in_
        self.n_samples_fit_ = X.shape[0]  # Used for default penalty calculation

        if self.mean is not None:
            if np.isscalar(self.mean):
                mean_arr = np.repeat(self.mean, X.shape[1])
            else:
                mean_arr = check_array(self.mean, ensure_2d=False)
            if mean_arr.shape[0] != X.shape[1]:
                raise ValueError(
                    f"mean must have {X.shape[1]} features, got {mean_arr.shape[0]}"
                )
            self._mean = mean_arr

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute data for L2 cost evaluation.

        For simple usage, just validates. Could be extended with
        cumulative sums for more efficient evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Cached data with cumulative sums.
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
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            L2 costs for each interval and features.
        """
        check_is_fitted(self)

        interval_specs = check_array(
            interval_specs,
            ensure_2d=True,
            ensure_min_features=self.interval_specs_ncols,
        )
        starts = interval_specs[:, 0]
        ends = interval_specs[:, 1]

        sums, sums2 = cache["sums"], cache["sums2"]
        if self.mean is None:
            costs = l2_cost_optim(starts, ends, sums, sums2)
        else:
            costs = l2_cost_fixed(starts, ends, sums, sums2, self._mean)

        return costs

    def get_default_penalty(self) -> float:
        """Get default penalty value for L2 cost.

        Returns
        -------
        float
            Default penalty value for L2 cost.
        """
        check_is_fitted(self)
        return make_bic_penalty(self.n_features_in_, self.n_samples_fit_)
