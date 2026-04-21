"""Continuous linear trend change score."""

__author__ = ["johannvk"]

from numbers import Integral

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseChangeScore
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import Interval
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def _lin_reg_cont_piecewise_linear_trend_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate the continuous linear trend score using linear regression.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    times : np.ndarray
        Time steps corresponding to the data points.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    scores = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, split, end = starts[i], splits[i], ends[i]
        split_interval_trend_data = np.zeros((end - start, 3))
        split_interval_trend_data[:, 0] = 1.0  # Intercept

        # Whole interval slope:
        split_interval_trend_data[:, 1] = times[start:end]  # Time steps

        # Change in slope from the 'split' index:
        split_interval_trend_data[(split - start) :, 2] = (
            times[split:end] - times[split]
        )

        # Calculate the slope and intercept for the whole interval:
        split_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data, X[start:end, :]
        )
        if (end - start) == 3 and split_interval_linreg_res[2] == 3:
            # If the interval is only 3 points long (the minimum),
            # and the model matrix is full rank, the residuals are zero.
            split_interval_squared_residuals = np.zeros((n_columns,))
        else:
            split_interval_squared_residuals = split_interval_linreg_res[1]

        # By only regressing onto the first two columns, we calculate the cost
        # without allowing for a change in slope at the split point.
        joint_interval_linreg_res = np.linalg.lstsq(
            split_interval_trend_data[:, np.array([0, 1])], X[start:end, :]
        )
        joint_interval_squared_residuals = joint_interval_linreg_res[1]

        # If either of the linear regression solutions failed, return NaN.
        if (len(split_interval_squared_residuals) == 0) or (
            len(joint_interval_squared_residuals) == 0
        ):
            scores[i, :] = np.nan
        else:
            scores[i, :] = (
                joint_interval_squared_residuals - split_interval_squared_residuals
            )

    return scores


@njit
def _continuous_piecewise_linear_trend_squared_contrast(
    signal: np.ndarray,
    first_interval_inclusive_start: int,
    second_interval_inclusive_start: int,
    non_inclusive_end: int,
) -> float:
    """Compute the squared contrast for a continuous piecewise linear trend.

    Analytical formulation from [1]_.

    Parameters
    ----------
    signal : np.ndarray
        1D signal data for a single feature.
    first_interval_inclusive_start : int
        Inclusive start of the full interval.
    second_interval_inclusive_start : int
        Inclusive start of the second sub-interval (the split point).
    non_inclusive_end : int
        Exclusive end of the full interval.

    Returns
    -------
    float
        Squared contrast value.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
       detection of multiple change points and change-point-like features. Journal of
       the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.
    """
    assert (
        first_interval_inclusive_start
        < second_interval_inclusive_start
        < non_inclusive_end - 1
    )

    # Translate named parameters to NOT-paper syntax.
    # We are zero-indexing the data, whilst the paper is one-indexing.
    s = first_interval_inclusive_start - 1
    e = non_inclusive_end - 1

    # Add one to NOT-syntax split index to account for the difference
    # in definition of where the change in slope starts from.
    b = second_interval_inclusive_start - 1 + 1

    l = e - s
    alpha = np.sqrt(
        6.0 / (l * (l**2 - 1) * (1 + (e - b + 1) * (b - s) + (e - b) * (b - s - 1)))
    )
    beta = np.sqrt(((e - b + 1.0) * (e - b)) / ((b - s - 1.0) * (b - s)))

    first_interval_slope = 3.0 * (b - s) + (e - b) - 1.0
    first_interval_constant = b * (e - s - 1.0) + 2.0 * (s + 1.0) * (b - s)

    second_interval_slope = 3.0 * (e - b) + (b - s) + 1.0
    second_interval_constant = b * (e - s - 1.0) + 2.0 * e * (e - b + 1)

    contrast = 0.0
    for t in range(s + 1, b + 1):
        contrast += (
            alpha * beta * (first_interval_slope * t - first_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    for t in range(b + 1, e + 1):
        contrast += (
            (-alpha / beta) * (second_interval_slope * t - second_interval_constant)
        ) * signal[t - first_interval_inclusive_start]

    return np.square(contrast)


@njit
def _analytical_cont_piecewise_linear_trend_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    """Evaluate the continuous piecewise linear trend score analytically.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the first intervals (inclusive).
    splits : np.ndarray
        Split indices between the intervals (contained in second interval).
    ends : np.ndarray
        End indices of the second intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    scores = np.zeros((len(starts), X.shape[1]))
    for i in range(len(starts)):
        start, split, end = starts[i], splits[i], ends[i]
        for j in range(X.shape[1]):
            scores[i, j] = _continuous_piecewise_linear_trend_squared_contrast(
                X[start:end, j],
                first_interval_inclusive_start=start,
                second_interval_inclusive_start=split,
                non_inclusive_end=end,
            )

    return scores


class ContinuousLinearTrendScore(BaseChangeScore):
    """Continuous linear trend change score.

    Calculates the difference in squared error between observed data and:

    - a two-parameter linear trend across the whole interval, and
    - a three-parameter linear trend with a kink at the split point.

    Intended for use with the NOT segment selection method as developed by
    Baranowski et al. [1]_. Accessible within the ``SeededBinarySegmentation``
    change detector by passing ``selection_method="narrowest"``.

    By default, time steps are assumed to be evenly spaced. In this case, an
    analytical solution is used to calculate the score for each column in the
    data, as described in [1]_.

    Parameters
    ----------
    time_col : int or None, default=None
        Column index to use as time stamps for calculating the piecewise linear
        trends. If ``None``, time steps are assumed to be evenly spaced and the
        analytical formulation is used.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
       detection of multiple change points and change-point-like features. Journal of
       the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import ContinuousLinearTrendScore
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = ContinuousLinearTrendScore()
    >>> scorer.fit(X)
    ContinuousLinearTrendScore()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 25, 50], [50, 75, 100]]))
    """

    _parameter_constraints: dict = {
        "time_col": [Interval(Integral, 0, None, closed="left"), None],
    }

    def __init__(self, time_col: int | None = None):
        self.time_col = time_col

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as detecting continuous (kink) changes."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.linear_trend_segment = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum interval size (2: at least 1 sample on each side of the split)."""
        return 2

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the score by storing data and extracting time stamps if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : ContinuousLinearTrendScore
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)

        if self.time_col is not None:
            times = X[:, self.time_col].astype(np.float64)
            self._time_stamps_ = times - times[0]
            all_cols = np.arange(X.shape[1])
            data_cols = all_cols[all_cols != self.time_col]
            self._data_cols_ = data_cols
        else:
            self._time_stamps_ = None
            self._data_cols_ = None

        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Store data for segment-wise evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)

        if self._data_cols_ is not None:
            data = X[:, self._data_cols_]
            times = X[:, self.time_col].astype(np.float64)
            times = times - times[0]
        else:
            data = X
            times = None

        return {"X": data, "times": times}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the continuous linear trend score on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 3)
            Each row is ``[start, split, end)``.

        Returns
        -------
        scores : ndarray of shape (n_intervals, n_features)
            Score for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        splits = interval_specs[:, 1]
        ends = interval_specs[:, 2]

        X = cache["X"]
        times = cache["times"]

        if times is None:
            return _analytical_cont_piecewise_linear_trend_score(
                starts, splits, ends, X
            )
        else:
            return _lin_reg_cont_piecewise_linear_trend_score(
                starts, splits, ends, X, times
            )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty.

        The model has 3 free parameters per feature (intercept, slope, slope change).

        Returns
        -------
        float
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, n_params=3)
