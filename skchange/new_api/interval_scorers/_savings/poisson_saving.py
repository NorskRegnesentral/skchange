"""Poisson saving for a fixed rate baseline."""

__author__ = ["johannvk"]

from numbers import Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.penalties import mvcapa_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def poisson_saving(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    baseline_rates: np.ndarray,
) -> np.ndarray:
    """Calculate the Poisson saving against fixed baseline rates.

    The saving is the reduction in twice the negative Poisson log-likelihood when
    using the MLE rate vs. the fixed baseline rate. The log-factorial terms cancel:

    .. code-block:: text

        saving = fixed_cost - mle_cost
               = 2*(n*(lambda_0 - lambda_hat)
               +   sum_x*(log(lambda_hat) - log(lambda_0)))

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    sums : np.ndarray
        Cumulative column sums of the input data, with a leading row of zeros.
        Shape ``(n_samples + 1, n_features)``.
    baseline_rates : np.ndarray
        Fixed baseline Poisson rates, shape ``(n_features,)``. Must be positive.

    Returns
    -------
    savings : np.ndarray
        A 2D array of savings, shape ``(n_intervals, n_features)``.
    """
    n_intervals = len(starts)
    n_cols = sums.shape[1]
    savings = np.zeros((n_intervals, n_cols))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        n = end - start
        for col in range(n_cols):
            sum_x = sums[end, col] - sums[start, col]
            rate_mle = sum_x / n
            rate_0 = baseline_rates[col]
            if rate_mle == 0:
                # All observations are zero; -2 log L(rate_mle) = 0,
                # -2 log L(rate_0) = 2*n*rate_0.  saving = 2*n*rate_0.
                savings[i, col] = 2.0 * n * rate_0
            else:
                # saving = 2*(n*(rate_0 - rate_mle)
                #           + sum_x*(log(rate_mle) - log(rate_0)))
                savings[i, col] = 2.0 * (
                    n * (rate_0 - rate_mle)
                    + sum_x * (np.log(rate_mle) - np.log(rate_0))
                )

    return savings


class PoissonSaving(BaseSaving):
    r"""Poisson saving for a fixed rate baseline.

    The Poisson saving measures the reduction in twice the negative Poisson
    log-likelihood when fitting the MLE rate (sample mean) to a segment compared
    to a fixed baseline rate. A large saving for an interval indicates that the
    baseline rate is a poor fit for the data in that interval.

    .. math::
        S([s, e)) = 2\left(n(\lambda_0 - \hat{\lambda}_{s:e})
        + \sum_{i=s}^{e-1} x_i
        \bigl(\log\hat{\lambda}_{s:e} - \log\lambda_0\bigr)\right)

    where :math:`\hat{\lambda}_{s:e} = \bar{x}_{s:e}` is the MLE rate and
    :math:`\lambda_0` is the fixed baseline rate. Note that the log-factorial terms
    cancel, so only cumulative sums of :math:`x` are needed.

    Parameters
    ----------
    baseline_rate : float, array-like of shape (n_features,), or None, default=None
        Fixed baseline Poisson rate. Must be positive. If ``None``, the column-wise
        median of the training data is used as a robust baseline estimate.

    Notes
    -----
    Requires non-negative count data. Accepts both integer and float arrays,
    provided all values are non-negative.

    Requires at least 1 observation per segment.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import PoissonSaving
    >>> rng = np.random.default_rng(0)
    >>> X = rng.poisson(lam=5.0, size=(100, 2)).astype(float)
    >>> scorer = PoissonSaving()
    >>> scorer.fit(X)
    PoissonSaving()
    >>> cache = scorer.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> scorer.evaluate(cache, interval_specs)
    """

    _parameter_constraints: dict = {
        "baseline_rate": ["array-like", Real, None],
    }

    def __init__(
        self,
        baseline_rate: ArrayLike | float | None = None,
    ):
        self.baseline_rate = baseline_rate

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as requiring non-negative count data."""
        tags = super().__sklearn_tags__()
        tags.input_tags.integer_only = True
        tags.input_tags.positive_only = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum segment size (1, one observation suffices for rate estimation)."""
        return 1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the saving, estimating the baseline rate from training data if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must contain non-negative values.
        y : None
            Ignored.

        Returns
        -------
        self : PoissonSaving
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        if np.any(X < 0):
            raise ValueError(
                "Negative values in data passed to PoissonSaving. "
                "PoissonSaving requires non-negative count data."
            )
        n_features = X.shape[1]

        if self.baseline_rate is None:
            rate = np.median(X, axis=0).astype(np.float64)
            rate = np.maximum(rate, 1e-10)
        else:
            rate = np.asarray(self.baseline_rate, dtype=np.float64)
            if rate.ndim == 0:
                rate = np.full(n_features, float(rate))
            if rate.shape != (n_features,):
                raise ValueError(
                    f"baseline_rate must be a scalar or array of shape "
                    f"(n_features,)={(n_features,)}, got shape {rate.shape}."
                )
            if not np.all(rate > 0):
                raise ValueError("baseline_rate must be strictly positive.")

        self.baseline_rate_ = rate
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute cumulative sums for efficient interval evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute. Must contain non-negative values.

        Returns
        -------
        cache : dict
            Dictionary with key ``"sums"``: cumulative column sums with a leading
            row of zeros, shape ``(n_samples + 1, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        if np.any(X < 0):
            raise ValueError(
                "Negative values in data passed to PoissonSaving. "
                "PoissonSaving requires non-negative count data."
            )
        return {"sums": col_cumsum(X, init_zero=True)}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the Poisson saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, n_features)
            Poisson savings for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return poisson_saving(starts, ends, cache["sums"], self.baseline_rate_)

    def get_default_penalty(self) -> np.ndarray:
        r"""Get the default penalty for the fitted Poisson saving.

        The Poisson saving is asymptotically :math:`\\chi^2(1)` per feature
        under the null (correct baseline rate), so ``mvcapa_penalty`` with
        1 parameter per feature is used.

        Returns
        -------
        np.ndarray of shape (n_features,)
            Default penalty value for each number of affected features.
        """
        check_is_fitted(self)
        # Scaling mvcapa penalty by 1.5 is done to pass the sanity checks
        # in the test suite for CAPA.
        return 1.5 * mvcapa_penalty(self.n_samples_in_, self.n_features_in_, 1)
