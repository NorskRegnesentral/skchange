"""Linear trend saving for a fixed slope/intercept baseline.

This module contains the LinearTrendSaving class, which is a saving function for
change point detection based on the reduction in squared error when fitting a
segment-wise best-fit linear trend versus a fixed baseline trend.
"""

__author__ = ["johannvk"]

from numbers import Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.interval_scorers._costs.linear_trend_cost import (
    fit_indexed_linear_trend,
)
from skchange.new_api.penalties import chi2_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def linear_trend_saving_index(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    baseline_params: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear trend saving using per-segment index time steps.

    Assumes that each segment's time steps are [0, 1, 2, ..., n-1],
    where n is the length of the segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    baseline_params : np.ndarray
        Fixed baseline trend parameters of shape (n_columns, 2), where each row
        is ``[slope, intercept]`` for the corresponding column.

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings, shape (n_intervals, n_columns).
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    savings = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        n = end - start
        segment_ts = np.arange(n, dtype=np.float64)
        for col in range(n_columns):
            segment_data = X[start:end, col]
            # Fixed-coefficient RSS:
            slope_0, intercept_0 = baseline_params[col, :]
            r_fixed = segment_data - (intercept_0 + slope_0 * segment_ts)
            fixed_rss = np.sum(r_fixed * r_fixed)
            # OLS RSS (indexed):
            slope_mle, intercept_mle = fit_indexed_linear_trend(segment_data)
            r_mle = segment_data - (intercept_mle + slope_mle * segment_ts)
            mle_rss = np.sum(r_mle * r_mle)
            savings[i, col] = fixed_rss - mle_rss

    return savings


class LinearTrendSaving(BaseSaving):
    r"""Linear trend saving for a fixed slope/intercept baseline.

    The saving measures the reduction in ordinary least-squares (OLS) residual
    sum of squares (RSS) when using segment-wise best-fit linear trend coefficients
    versus fixed baseline (slope, intercept) pairs:

    .. math::
        S_j([s, e)) = \text{RSS}(\beta_0^{(j)};\, X_{s:e,j})
                    - \min_\beta\,\text{RSS}(\beta;\, X_{s:e,j})

    where :math:`\beta_0^{(j)} = (\text{slope}_j, \text{intercept}_j)` are the
    fixed baseline parameters for column :math:`j`. A large saving indicates the
    baseline trend is a poor fit for the segment.

    By default the time steps are assumed to be ``[0, 1, ..., (end-start)-1]``
    within each segment.

    Inspired by [1]_ who propose the same cost function for detecting changes in
    piecewise-linear signals.

    Parameters
    ----------
    baseline_slope : float or array-like of shape (n_features,), default=0
        Fixed baseline slope for each value column. If a scalar, the same slope is
        used for all columns.
    baseline_intercept : float, array-like of shape (n_features,), or None,\
 default=None
        Fixed baseline intercept for each value column. If a scalar, the same
        intercept is used for all columns. If ``None``, the OLS intercept fitted
        on the training data is used.

    References
    ----------
    .. [1] Fearnhead, P., & Grose, D. (2024). cpop: Detecting Changes in \
    Piecewise-Linear Signals. Journal of Statistical Software, 109(7), 1-30.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LinearTrendSaving
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(100, 2))
    >>> scorer = LinearTrendSaving()
    >>> scorer.fit(X)
    LinearTrendSaving()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    _parameter_constraints: dict = {
        "baseline_slope": ["array-like", Real],
        "baseline_intercept": ["array-like", Real, None],
    }

    def __init__(
        self,
        baseline_slope: ArrayLike | float = 0,
        baseline_intercept: ArrayLike | float | None = None,
    ):
        self.baseline_slope = baseline_slope
        self.baseline_intercept = baseline_intercept

    def __sklearn_tags__(self):
        """Return tags marking this saving as requiring linear-trend segment data."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.linear_trend_segment = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        For linear trend fitting, we need at least 3 points.
        (2 points always give a perfect fit with zero residuals.)

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 3

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the saving, estimating baseline trend if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : LinearTrendSaving
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        n_features = X.shape[1]

        self.value_cols_ = list(range(n_features))
        X_values = X[:, self.value_cols_]

        # Default baseline estimation:
        #   - slope  defaults to 0 (no-trend prior)
        #   - intercept defaults to the column-wise median
        #
        # Any slope estimator (OLS, Huber, Theil-Sen) is biased when the training
        # window contains a changed segment at the end, because the shifted points
        # carry a consistent directional signal.  A slope of 0 with a robust
        # location (median) is the correct default for IID or piecewise-constant
        # data; users with a genuine baseline trend supply the parameters explicitly.
        default_intercepts = np.median(X_values, axis=0)

        self.baseline_slope_ = self._resolve_param(
            self.baseline_slope, np.zeros(len(self.value_cols_)), "baseline_slope"
        )
        self.baseline_intercept_ = self._resolve_param(
            self.baseline_intercept, default_intercepts, "baseline_intercept"
        )
        # Pack into (n_value_cols_, 2) array for numba kernels: [slope, intercept]
        self._baseline_params = np.column_stack(
            [self.baseline_slope_, self.baseline_intercept_]
        )
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
            Dictionary with key:

            - ``"values"``: 2D data array of shape ``(n_samples, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {"values": np.ascontiguousarray(X[:, self.value_cols_])}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate linear trend saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from ``precompute()``.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_value_cols)
            RSS saving (fixed - OLS) for each interval and trend column.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        values = cache["values"]
        return linear_trend_saving_index(starts, ends, values, self._baseline_params)

    def get_default_penalty(self) -> float:
        r"""Get the default penalty for the fitted linear trend saving.

        The saving per column is asymptotically :math:`\chi^2(2)` under the null
        (correct baseline slope and intercept), so ``chi2_penalty`` with 2 degrees
        of freedom per column and ``n_value_cols_`` columns is used.

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        # 2 d.o.f. per column: slope and intercept
        return chi2_penalty(self.n_samples_in_, 2 * len(self.value_cols_))

    def _resolve_param(
        self,
        param: ArrayLike | float | None,
        ols_values: np.ndarray,
        name: str,
    ) -> np.ndarray:
        """Resolve a baseline parameter to shape (n_value_cols_,).

        Parameters
        ----------
        param : array-like, float, or None
            User-provided parameter value.
        ols_values : np.ndarray of shape (n_value_cols_,)
            OLS estimates to fall back to when param is None.
        name : str
            Parameter name for error messages.

        Returns
        -------
        np.ndarray of shape (n_value_cols_,)
        """
        if param is None:
            return ols_values.copy()
        arr = np.asarray(param, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(len(self.value_cols_), float(arr))
        if arr.shape != (len(self.value_cols_),):
            raise ValueError(
                f"{name} must be a scalar or array of shape "
                f"({len(self.value_cols_)},), got shape {arr.shape}."
            )
        return arr
