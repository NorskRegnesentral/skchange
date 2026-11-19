"""Continuous linear trend change score."""

__author__ = ["johannvk"]

from numbers import Integral

import numpy as np

from skchange.interval_scorers._base import BaseChangeScore
from skchange.penalties import bic_penalty
from skchange.types import ArrayLike
from skchange.utils._numba import njit
from skchange.utils._param_validation import Interval, _fit_context
from skchange.utils._tags import SkchangeTags
from skchange.utils.validation import (
    check_interval_specs,
    check_is_fitted,
    check_time_col,
    validate_data,
)


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def _cumulative_sum_trend_score_run(
    scores: np.ndarray,
    row_start: int,
    row_stop: int,
    splits: np.ndarray,
    start: int,
    end: int,
    X: np.ndarray,
    times: np.ndarray,
) -> None:
    """Score all splits of a single interval ``[start, end)`` in O(1) per split.

    With ``Z = [1, t]`` the no-kink design, ``h`` the kink regressor
    ``(t - t_split)_+`` and ``G = Z'Z``, the Frisch-Waugh-Lovell identity gives

        RSS_0 - RSS_1 = (h'x - c'G^{-1}g)^2 / (h'h - c'G^{-1}c),

    with ``c = Z'h`` and ``g = Z'x``.

    For numerical robustness, time is centred at the interval midpoint and the
    data at the interval mean. The kink sums ``h'h``, ``h'1``, ``h'u`` and
    ``h'x`` are accumulated from the right end, re-expressing the running sums
    relative to each new split time. They are then sums of non-negative
    kink values, and no differences of large cumulative sums are formed, which
    would lose precision for long intervals or intervals far into a long series.

    The rows ``row_start:row_stop`` of ``scores`` are filled.
    """
    n = end - start
    n_features = X.shape[1]
    t_mid = 0.5 * (times[start] + times[end - 1])

    u = np.empty(n)
    t1 = 0.0  # Sum of u.
    t2 = 0.0  # Sum of u^2.
    for i in range(n):
        u[i] = times[start + i] - t_mid
        t1 += u[i]
        t2 += u[i] * u[i]
    det_g = n * t2 - t1 * t1

    # Kink sums that depend only on the time stamps, for each split position k.
    h_sum = np.empty(n)  # 1'h
    h_u = np.empty(n)  # u'h
    h_h = np.empty(n)  # h'h
    # Running sums over i >= k of (u_i - u_k)^a, a = 0, 1, 2.
    count = 0.0
    r1 = 0.0
    r2 = 0.0
    for k in range(n - 1, -1, -1):
        if count > 0.0:
            d = u[k + 1] - u[k]
            r2 += d * (2.0 * r1 + count * d)
            r1 += count * d
        count += 1.0
        h_sum[k] = r1
        h_u[k] = r2 + u[k] * r1
        h_h[k] = r2

    x_c = np.empty(n)
    for j in range(n_features):
        x_mean = 0.0
        for i in range(n):
            x_mean += X[start + i, j]
        x_mean /= n

        g0 = 0.0  # 1'x
        g1 = 0.0  # u'x
        for i in range(n):
            x_c[i] = X[start + i, j] - x_mean
            g0 += x_c[i]
            g1 += u[i] * x_c[i]
        # G^{-1} Z'x via the closed-form 2x2 inverse.
        ginv_g0 = (t2 * g0 - t1 * g1) / det_g
        ginv_g1 = (n * g1 - t1 * g0) / det_g

        # h'x for each split k, with running sums of x and (u_i - u_k) x over i >= k.
        h_x = np.empty(n)
        sx = 0.0
        sdx = 0.0
        for k in range(n - 1, -1, -1):
            if k < n - 1:
                sdx += (u[k + 1] - u[k]) * sx
            sx += x_c[k]
            h_x[k] = sdx

        for row in range(row_start, row_stop):
            k = splits[row] - start
            c0 = h_sum[k]
            c1 = h_u[k]
            hh = h_h[k]

            ginv_c0 = (t2 * c0 - t1 * c1) / det_g
            ginv_c1 = (n * c1 - t1 * c0) / det_g
            denominator = hh - (c0 * ginv_c0 + c1 * ginv_c1)
            numerator = h_x[k] - (c0 * ginv_g0 + c1 * ginv_g1)

            # Splits whose kink cannot be told apart from a straight line, to
            # within roundoff, are not scored.
            if denominator > 1e-14 * hh:
                scores[row, j] = numerator * numerator / denominator
            else:
                scores[row, j] = 0.0


@njit(cache=True)
def _cumulative_sum_cont_piecewise_linear_trend_score(
    starts: np.ndarray,
    splits: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Evaluate the continuous linear trend score with cumulative sums.

    Rows with the same ``(start, end)`` that are adjacent share one pass over the
    interval, after which each split costs O(1) per feature. Other rows form runs
    of length one and cost O(interval length), as in a direct evaluation.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    splits : np.ndarray
        Split indices (contained in the second part).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    times : np.ndarray
        Time stamps corresponding to the data points.

    Returns
    -------
    scores : np.ndarray
        A 2D array of scores. One row for each interval.
    """
    n_rows = len(starts)
    scores = np.zeros((n_rows, X.shape[1]))

    row = 0
    while row < n_rows:
        run_stop = row + 1
        while (
            run_stop < n_rows
            and starts[run_stop] == starts[row]
            and ends[run_stop] == ends[row]
        ):
            run_stop += 1

        _cumulative_sum_trend_score_run(
            scores, row, run_stop, splits, starts[row], ends[row], X, times
        )
        row = run_stop

    return scores


class ContinuousLinearTrendScore(BaseChangeScore):
    r"""Continuous linear trend change score.

    Calculates the difference in squared error between observed data and:

    - a two-parameter linear trend across the whole interval, and
    - a three-parameter linear trend with a kink at the split point.

    Intended for use with the NOT segment selection method as developed by
    Baranowski et al. [1]_. Accessible within the ``SeededBinarySegmentation``
    change detector by passing ``selection_method="narrowest"``.

    The score is computed from sums of the form :math:`\sum t^a x`, :math:`a=0,1`,
    and :math:`\sum t^a`, :math:`a=0,1,2`, within the interval. Evaluating
    many splits of the same interval in one call (consecutive rows with equal
    ``start`` and ``end``) costs O(1) per split after one pass over the
    interval, as in ``SeededBinarySegmentation``. The result is identical to the
    analytical formulation of [1]_ for evenly spaced time steps, and also holds for
    irregular time stamps.

    Parameters
    ----------
    time_col : int or None, default=None
        Column index to use as time stamps for calculating the piecewise linear
        trends. If ``None``, time steps are assumed to be evenly spaced. The column
        must be strictly monotonically increasing.

        .. experimental::
            Support for ``time_col`` is experimental and the API may change.

    References
    ----------
    .. [1] Baranowski, R., Chen, Y., & Fryzlewicz, P. (2019). Narrowest-over-threshold
       detection of multiple change points and change-point-like features. Journal of
       the Royal Statistical Society Series B: Statistical Methodology, 81(3), 649-672.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.interval_scorers import ContinuousLinearTrendScore
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
        tags.input_tags.timestamps = self.time_col is not None
        return tags

    @property
    def min_size(self) -> int:
        """Minimum interval size (2: at least 1 sample on each side of the split)."""
        return 2

    @_fit_context(prefer_skip_nested_validation=True)
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
            check_time_col(X, self.time_col, self.__class__.__name__)
            all_cols = np.arange(X.shape[1])
            self.value_cols_ = all_cols[all_cols != self.time_col]
        else:
            self.value_cols_ = list(range(X.shape[1]))

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

        data = X[:, self.value_cols_]
        if self.time_col is not None:
            times = X[:, self.time_col].astype(np.float64)
            times = times - times[0]
        else:
            times = np.arange(X.shape[0], dtype=np.float64)

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

        return _cumulative_sum_cont_piecewise_linear_trend_score(
            starts, splits, ends, cache["X"], cache["times"]
        )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty.

        The model has 3 free parameters per feature (intercept, slope, slope change).

        Returns
        -------
        float
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, 3 * self.n_features_in_)
