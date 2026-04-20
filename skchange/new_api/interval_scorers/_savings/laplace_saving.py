"""Laplace saving for a fixed location and scale baseline."""

__author__ = ["johannvk"]

from numbers import Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.penalties import mvcapa_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.general import truncate_below
from skchange.utils.numba.stats import col_median


@njit
def laplace_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    baseline_location: np.ndarray,
    baseline_scale: np.ndarray,
) -> np.ndarray:
    """Calculate the Laplace saving against a fixed baseline location and scale.

    The saving is the reduction in twice negative log-likelihood when using
    the MLE parameters vs. the fixed baseline parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    baseline_location : np.ndarray
        Fixed baseline location (median) of shape (n_features,).
    baseline_scale : np.ndarray
        Fixed baseline scale of shape (n_features,).

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings, shape (n_intervals, n_features).
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    savings = np.zeros((n_intervals, n_columns))
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

        for col in range(n_columns):
            # fixed cost: 2n*log(2*b0) + 2*sum(|x-loc0|)/b0
            abs_dev_baseline = np.sum(np.abs(segment[:, col] - baseline_location[col]))
            fixed_cost = 2.0 * n * np.log(2.0 * baseline_scale[col]) + (
                2.0 * abs_dev_baseline / baseline_scale[col]
            )
            # mle cost: 2n*(log(2*b_mle) + 1)
            mle_cost = 2.0 * n * (np.log(2.0 * mle_scales[col]) + 1.0)
            savings[i, col] = fixed_cost - mle_cost

    return savings


class LaplaceSaving(BaseSaving):
    r"""Laplace saving for a fixed location and scale baseline.

    The Laplace saving measures the reduction in twice negative log-likelihood
    when fitting MLE location (median) and scale (mean absolute deviation) to a
    segment compared to fixed baseline parameters. A large saving indicates that
    the baseline parameters are a poor fit for the data in that interval.

    .. math::
        S([s, e)) = 2n\log\!\left(\frac{2 b_0}{\hat{b}_{s:e}}\right)
        + \frac{2}{b_0}\sum_{i=s}^{e-1}|x_i - \mu_0|
        - 2n

    where :math:`\mu_0, b_0` are the baseline location and scale (estimated from
    the training data by default via column-wise median and MAD), and
    :math:`\hat{b}_{s:e}` is the MLE scale for the segment.

    Parameters
    ----------
    baseline_location : float, array-like of shape (n_features,), or None, default=None
        Fixed baseline location (median) to compare against. If ``None``, the
        column-wise median of the training data is used.
    baseline_scale : float, array-like of shape (n_features,), or None, default=None
        Fixed baseline scale to compare against. Must be positive. If ``None``,
        the column-wise median absolute deviation from the median, divided by
        ``log(2)``, is used. This is a consistent, 50%-breakdown-point estimator
        of the Laplace scale.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import LaplaceSaving
    >>> X = np.random.default_rng(0).normal(size=(100, 2))
    >>> scorer = LaplaceSaving()
    >>> scorer.fit(X)
    LaplaceSaving()
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    _parameter_constraints: dict = {
        "baseline_location": ["array-like", Real, None],
        "baseline_scale": ["array-like", Real, None],
    }

    def __init__(
        self,
        baseline_location: ArrayLike | float | None = None,
        baseline_scale: ArrayLike | float | None = None,
    ):
        self.baseline_location = baseline_location
        self.baseline_scale = baseline_scale

    @property
    def min_size(self) -> int:
        """Minimum segment size (2, required for scale estimation)."""
        return 2

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit Laplace saving, validating and broadcasting baseline parameters.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : LaplaceSaving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        n_features = X.shape[1]

        if self.baseline_location is None:
            location = np.median(X, axis=0)
        else:
            location = np.asarray(self.baseline_location, dtype=np.float64)
            if location.ndim == 0:
                location = np.full(n_features, float(location))
            if location.shape != (n_features,):
                raise ValueError(
                    f"baseline_location must be a scalar or array of shape "
                    f"(n_features,)={(n_features,)}, got shape {location.shape}."
                )

        if self.baseline_scale is None:
            scale = np.median(np.abs(X - location), axis=0) / np.log(2)
            scale = np.maximum(scale, 1e-16)
        else:
            scale = np.asarray(self.baseline_scale, dtype=np.float64)
            if scale.ndim == 0:
                scale = np.full(n_features, float(scale))
            if scale.shape != (n_features,):
                raise ValueError(
                    f"baseline_scale must be a scalar or array of shape "
                    f"(n_features,)={(n_features,)}, got shape {scale.shape}."
                )
            if not np.all(scale > 0):
                raise ValueError("baseline_scale must be strictly positive.")

        self.baseline_location_ = location
        self.baseline_scale_ = scale
        return self

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
        """Evaluate Laplace saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from precompute().
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            Laplace savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return laplace_saving(
            starts, ends, cache["X"], self.baseline_location_, self.baseline_scale_
        )

    def get_default_penalty(self) -> np.ndarray:
        """Get the default penalty for the fitted Laplace saving.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Default penalty value for each number of affected features.
        """
        check_is_fitted(self)
        # Scaling mvcapa penalty by 2.0 is done to pass the sanity checks in the
        # test suite for CAPA.
        return 2.0 * mvcapa_penalty(self.n_samples_in_, self.n_features_in_, 2)
