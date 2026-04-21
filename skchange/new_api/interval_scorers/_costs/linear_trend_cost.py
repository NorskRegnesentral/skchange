"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a best fit linear trend line within each interval.
"""

__author__ = ["johannvk"]

from numbers import Integral

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import Interval, _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def fit_linear_trend(time_steps: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Parameters
    ----------
    time_steps : np.ndarray
        1D array of time points.
    values : np.ndarray
        1D array of data points.

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    mean_t = np.mean(time_steps)
    centered_time_steps = time_steps - mean_t
    mean_value = np.mean(values)

    # Calculate linear regression denominator = sum((t-mean_t)²):
    denominator = np.sum(np.square(centered_time_steps))

    # Calculate linear regression numerator = sum((t-mean_t)*(x-mean_x)):
    numerator = np.sum(centered_time_steps * (values - mean_value))

    slope = numerator / denominator
    intercept = mean_value - slope * mean_t

    return slope, intercept


@njit
def fit_indexed_linear_trend(xs: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Assuming the time steps are [0, 1, 2, ..., n-1], we can optimize the calculation.

    Parameters
    ----------
    xs : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    n_samples = len(xs)

    # For evenly spaced time steps [0, 1, 2, ..., n-1],
    # the mean time step is (n-1)/2.
    mean_t = (n_samples - 1) / 2.0

    # Optimized calculation for denominator:
    # sum of (t - mean_t)^2 = n*(n^2-1)/12
    denominator = n_samples * (n_samples * n_samples - 1) / 12.0

    # Calculate numerator: sum((t-mean_t)*(x-mean_x))
    # numerator = np.sum((np.arange(n) - mean_t) * (xs - mean_x))
    mean_x = np.mean(xs)
    numerator = 0.0
    for i in range(n_samples):
        numerator += (i - mean_t) * (xs[i] - mean_x)

    slope = numerator / denominator
    intercept = mean_x - slope * mean_t

    return slope, intercept


@njit
def linear_trend_cost_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, times: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with optimized parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    times : np.ndarray
        Time points the data points were observed at.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment_times = times[start:end]
        for col in range(n_columns):
            segment_data = X[start:end, col]
            slope, intercept = fit_linear_trend(
                time_steps=segment_times, values=segment_data
            )
            costs[i, col] = np.sum(
                np.square(segment_data - (intercept + slope * segment_times))
            )

    return costs


@njit
def linear_trend_cost_index_times_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with optimized parameters.

    Assumes that the data is observed at time steps [0, 1, 2, ..., n-1]
    within each segment, with n being the length of the segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            segment_data = X[start:end, col]

            slope, intercept = fit_indexed_linear_trend(segment_data)
            linear_trend_values = intercept + slope * np.arange(segment_data.shape[0])
            costs[i, col] = np.sum(np.square(segment_data - linear_trend_values))

    return costs


class LinearTrendCost(BaseCost):
    """Linear trend cost function.

    This cost function calculates the sum of squared errors between data points
    and a linear trend line fitted to each interval. For each interval and each column,
    a straight line is fitted (or provided as a fixed parameter) and the squared
    differences between the actual data points and the fitted line are summed.

    By default the time steps are assumed to be [0, 1, 2, ..., (start - end) - 1] for
    each segment. If a time column is provided, time steps are taken from that column.

    Inspired by [1]_ who propose the same cost function for detecting changes in
    piecewise-linear signals, but within an optimization problem which enforces
    continuity of the linear trend across segments. To achieve similar results
    in the ``skchange`` package, we recommend using the ``ContinuousLinearTrendScore``
    within a change detection algorithm such as ``SeededBinarySegmentation`` or
    ``MovingWindow``.

    Parameters
    ----------
    time_col : int or None, default=None
        By default time steps are assumed to be evenly spaced with unit distance.
        If a time column index is provided, its values are used as the time steps for
        calculating the linear trends. The time column values must be convertible to
        ``float`` dtype.  The time column is excluded from the trend data and does not
        contribute to the cost. If your time column is of ``datetime.datetime`` type,
        convert it to a numeric column first.

    References
    ----------
    .. [1] Fearnhead, P., & Grose, D. (2024). cpop: Detecting Changes in \
    Piecewise-Linear Signals. Journal of Statistical Software, 109(7), 1-30.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LinearTrendCost
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> scorer = LinearTrendCost()
    >>> scorer.fit(X)
    LinearTrendCost()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    _parameter_constraints: dict = {
        "time_col": [Interval(Integral, 0, None, closed="left"), None],
    }

    def __init__(self, time_col: int | None = None):
        self.time_col = time_col

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this cost as requiring linear-trend segment data."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.linear_trend_segment = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        For linear trend fitting, we need at least 3 points to have a
        non-trivial residual sum of squares (2 points always give a perfect fit).

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 3

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the cost, storing time stamps and validating fixed parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If ``time_col`` is set, that column is used for
            time steps and excluded from the trend data.
        y : None
            Ignored.

        Returns
        -------
        self : LinearTrendCost
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        n_features = X.shape[1]

        if self.time_col is not None:
            if not (0 <= self.time_col < n_features):
                raise ValueError(
                    f"time_col={self.time_col} is out of range for data with "
                    f"{n_features} columns."
                )
            self.time_col_ = self.time_col
            self.value_cols_ = [c for c in range(n_features) if c != self.time_col_]
        else:
            self.time_col_ = None
            self.value_cols_ = list(range(n_features))

        self.n_value_cols_ = len(self.value_cols_)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Extract and cache trend data and time stamps.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with keys:

            - ``"values"``: 2D trend data of shape ``(n_samples, n_value_cols)``,
              with the time column removed if applicable.
            - ``"time_stamps"``: 1D float array of shape ``(n_samples,)``, or
              ``None`` if using per-segment index times.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        if self.time_col_ is not None:
            # Extract time stamps from this data and start at zero:
            time_stamps = X[:, self.time_col_].copy()
            time_stamps -= time_stamps[0]
        else:
            time_stamps = None
        return {
            "values": np.ascontiguousarray(X[:, self.value_cols_]),
            "time_stamps": time_stamps,
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate linear trend cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from ``precompute()``.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_value_cols)
            Sum of squared deviations from the linear trend for each interval
            and each trend column.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        values = cache["values"]
        time_stamps = cache["time_stamps"]

        if time_stamps is not None:
            return linear_trend_cost_mle(starts, ends, values, time_stamps)
        else:
            return linear_trend_cost_index_times_mle(starts, ends, values)

    def get_default_penalty(self) -> float:
        """Get the default penalty for the fitted linear trend cost.

        Each column requires 2 parameters (slope, intercept), so BIC is computed
        with ``2 * n_value_cols`` parameters.

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        # For each column we need 2 parameters: slope and intercept
        return bic_penalty(self.n_samples_in_, 2 * self.n_value_cols_)
