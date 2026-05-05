"""L2 saving for a fixed baseline mean."""

__author__ = ["Tveten"]

from numbers import Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.interval_scorers._savings._utils import (
    resolve_baseline_location,
)
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils.validation import (
    check_interval_specs,
    validate_data,
)
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def l2_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 saving for a zero-valued baseline mean.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.
    sums : np.ndarray
        Cumulative sum of the input data, with a row of 0-entries as the first row.

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate saving for the corresponding input data column.
    """
    n = (ends - starts).reshape(-1, 1)
    return (sums[ends] - sums[starts]) ** 2 / n


class L2Saving(BaseSaving):
    r"""L2 saving for a fixed baseline mean.

    The L2 saving measures the reduction in squared error when fitting an optimal
    mean to a segment compared to a fixed baseline mean. It is computed directly
    from cumulative sums, making evaluation O(1) per interval after a linear-time
    precomputation step.

    .. math::
        S([s, e)) = \frac{1}{e - s}\left(\sum_{i=s}^{e-1} (x_i - \mu_0)\right)^2

    where :math:`\mu_0` is the baseline mean.

    See [1]_ for theoretical background.

    Parameters
    ----------
    baseline_mean : float, array-like of shape (n_features,), or None, default=None
        Fixed baseline mean per feature. If ``None``, estimated as the column-wise
        median of the training data.

    See [1]_ for theoretical background.

    References
    ----------
    .. [1] Fisch, A. T., Eckley, I. A., & Fearnhead, P. (2022). A linear time method
       for the detection of collective and point anomalies. Statistical Analysis and
       DataMining: The ASA Data Science Journal, 15(4), 494-508.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import L2Saving
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = L2Saving()
    >>> scorer.fit(X)
    L2Saving()
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    _parameter_constraints: dict = {
        "baseline_mean": ["array-like", Real, None],
    }

    def __init__(self, baseline_mean: ArrayLike | float | None = None):
        self.baseline_mean = baseline_mean

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit L2 saving, estimating the baseline mean if not provided.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : L2Saving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        self.baseline_mean_ = resolve_baseline_location(self.baseline_mean, X)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for fast interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key ``"sums"``: cumulative column sums of
            ``(X - baseline_mean_)`` with a leading row of zeros,
            shape ``(n_samples + 1, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        return {"sums": col_cumsum(X - self.baseline_mean_, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate L2 saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            L2 savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return l2_saving(starts, ends, cache["sums"])
