"""Laplace distribution (twice negative log-likelihood) cost function."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_median


@njit
def laplace_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Calculate the Laplace twice negative log-likelihood cost for each segment.

    Fits the optimal (MLE) location (median) and scale (mean absolute deviation
    from median) per segment.

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
        A 2D array of costs, shape (n_intervals, n_features).
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))
    mle_locations = np.zeros(n_columns)
    mle_scales = np.zeros(n_columns)

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        n = end - start
        segment = X[start:end]
        mle_locations = col_median(segment, output_array=mle_locations)
        for col in range(n_columns):
            mle_scales[col] = np.mean(np.abs(segment[:, col] - mle_locations[col]))
        mle_scales = truncate_below(mle_scales, 1e-16)
        # twice negative log-likelihood: 2n*log(2*scale) + 2*sum(|x-loc|)/scale
        # At MLE scale = mean(|x-loc|), so sum/scale = n, giving 2n*log(2*scale) + 2n
        for col in range(n_columns):
            costs[i, col] = 2.0 * n * (np.log(2.0 * mle_scales[col]) + 1.0)

    return costs


class LaplaceCost(BaseCost):
    r"""Laplace distribution twice negative log-likelihood cost.

    Computes the twice negative Laplace log-likelihood for each segment, fitting
    the optimal (MLE) location (median) and scale (mean absolute deviation) per
    segment.

    .. math::
        C(X_{s:e}) = 2n \left(\log(2\hat{b}_{s:e}) + 1\right)

    where :math:`\hat{b}_{s:e} = \frac{1}{n}\sum_{i=s}^{e-1}|x_i -
    \text{median}(X_{s:e})|` is the MLE scale of the segment.

    Notes
    -----
    Requires at least 2 observations per segment for scale estimation.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LaplaceCost
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> cost = LaplaceCost()
    >>> cost.fit(X)
    LaplaceCost()
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> cost.evaluate(cache, interval_specs)
    """

    @property
    def min_size(self) -> int:
        """Minimum segment size (2, required for scale estimation)."""
        return 2

    def precompute(self, X: ArrayLike) -> dict:
        """Store data for interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key ``"X"``: the data array.
        """
        # The MLE location of the Laplace distribution is the segment median.
        # The median is tricky to precompute, so it is computed on the fly in evaluate.
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {"X": X}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate Laplace cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            Laplace costs for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return laplace_cost(starts, ends, cache["X"])

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted Laplace cost.

        The Laplace cost has 2 parameters per feature (location and scale).

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        return 1.5 * bic_penalty(self.n_samples_in_, 2 * self.n_features_in_)
