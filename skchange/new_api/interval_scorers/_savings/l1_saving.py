"""L1 saving for a fixed median baseline."""

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_median


@njit
def l1_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    locations: np.ndarray,
) -> np.ndarray:
    """Calculate the L1 saving against a fixed baseline location.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    locations : np.ndarray
        Fixed baseline location of shape (n_features,).

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings. One row for each interval. The number of columns
        is equal to the number of columns in the input data, where each column
        represents the univariate saving for the corresponding input data column.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    savings = np.zeros((n_intervals, n_columns))
    medians = np.zeros(n_columns)

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]
        medians = col_median(segment, output_array=medians)
        savings[i, :] = np.sum(
            np.abs(segment - locations) - np.abs(segment - medians),
            axis=0,
        )

    return savings


class L1Saving(BaseSaving):
    r"""L1 saving for a fixed median baseline.

    The L1 saving measures the reduction in absolute error when fitting an optimal
    median to a segment compared to a fixed baseline location. A large saving for
    an interval indicates that the baseline location is a poor fit for the data
    in that interval.

    .. math::
        S([s, e)) = \sum_{i=s}^{e-1} |x_i - \text{location}|
            - \sum_{i=s}^{e-1} |x_i - \text{median}(X_{s:e})|

    Note that this saving assumes a **fixed baseline location** provided at fit time.
    The data should be centred by subtracting the baseline before fitting, so that
    the baseline becomes zero.

    Parameters
    ----------
    baseline_location : float or array-like of shape (n_features,), default=0.0
        Fixed baseline location (median) to compare against the optimal median.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import L1Saving
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = L1Saving()
    >>> scorer.fit(X)
    L1Saving()
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    def __init__(self, baseline_location: ArrayLike | float = 0.0):
        self.baseline_location = baseline_location

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit L1 saving, validating and broadcasting the baseline location.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : L1Saving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        location = np.asarray(self.baseline_location, dtype=np.float64)
        if location.ndim == 0:
            location = np.full(X.shape[1], location)
        if location.shape != (X.shape[1],):
            raise ValueError(
                f"baseline_location must be a scalar or array of shape "
                f"(n_features,)={(X.shape[1],)}, got shape {location.shape}."
            )
        self.baseline_location_ = location
        return self

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate L1 saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            L1 savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return l1_saving(starts, ends, cache["X"], self.baseline_location_)
