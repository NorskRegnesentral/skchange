"""Gaussian (negative log-likelihood) cost function."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_cumsum


@njit
def gaussian_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian negative log-likelihood cost for each segment.

    Fits the optimal (MLE) mean and variance per segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : np.ndarray
        Cumulative sum of the squared input data, with a row of 0-entries as the
        first row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate cost for the corresponding input data column.
    """
    n = (ends - starts).reshape(-1, 1)
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    var = partial_sums2 / n - (partial_sums / n) ** 2
    var = truncate_below(var, 1e-16)  # standard deviation lower bound of 1e-8
    return n * np.log(2 * np.pi * var) + n


class GaussianCost(BaseCost):
    r"""Gaussian (negative log-likelihood) cost function.

    Computes the negative Gaussian log-likelihood for each segment, fitting the
    optimal (MLE) mean and variance per segment.

    .. math::
        C(X_{s:e}) = n \log(2\pi\hat{\sigma}^2_{s:e}) + n

    where :math:`\hat{\sigma}^2_{s:e}` is the MLE variance of the segment.

    Notes
    -----
    Requires at least 2 observations per segment for variance estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import GaussianCost
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> cost = GaussianCost()
    >>> cost.fit(X)
    GaussianCost()
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> cost.evaluate(cache, interval_specs)
    """

    @property
    def min_size(self) -> int:
        """Minimum segment size (2, required for variance estimation)."""
        return 2

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for efficient interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with keys ``"sums"`` and ``"sums2"``: cumulative column sums
            and cumulative column sums of squares, each with a leading row of zeros,
            shape ``(n_samples + 1, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {
            "sums": col_cumsum(X, init_zero=True),
            "sums2": col_cumsum(X**2, init_zero=True),
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate Gaussian cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            Gaussian costs for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return gaussian_cost(starts, ends, cache["sums"], cache["sums2"])

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted Gaussian cost.

        The Gaussian cost has 2 parameters per feature (mean and variance).

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, 2 * self.n_features_in_)
