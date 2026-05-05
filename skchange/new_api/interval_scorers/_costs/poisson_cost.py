"""Poisson distribution (twice negative log-likelihood) cost function."""

__author__ = ["johannvk"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum


@njit
def poisson_cost(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    log_factorial_sums: np.ndarray,
) -> np.ndarray:
    """Calculate the Poisson twice negative log-likelihood cost for each segment.

    Fits the optimal (MLE) rate (sample mean) per segment.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the segments (inclusive).
    ends : np.ndarray
        End indices of the segments (exclusive).
    sums : np.ndarray
        Cumulative column sums of the input data, with a leading row of zeros.
        Shape ``(n_samples + 1, n_features)``.
    log_factorial_sums : np.ndarray
        Cumulative column sums of ``log(x!)``, with a leading row of zeros.
        Shape ``(n_samples + 1, n_features)``.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs, shape ``(n_intervals, n_features)``. Each entry is
        twice the negative Poisson log-likelihood at the MLE rate.
    """
    n_intervals = len(starts)
    n_cols = sums.shape[1]
    costs = np.zeros((n_intervals, n_cols))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        n = end - start
        for col in range(n_cols):
            sum_x = sums[end, col] - sums[start, col]
            sum_log_fac = log_factorial_sums[end, col] - log_factorial_sums[start, col]
            rate = sum_x / n  # MLE rate = sample mean
            if rate > 0:
                # -2 log L(rate_mle) = 2*(n*rate*(1 - log(rate)) + sum(log(x_i!)))
                costs[i, col] = 2.0 * (n * rate * (1.0 - np.log(rate)) + sum_log_fac)
            # else: rate == 0 → all x_i = 0 → cost = 0 (already initialised to 0)

    return costs


class PoissonCost(BaseCost):
    r"""Poisson distribution twice negative log-likelihood cost.

    Computes twice the negative Poisson log-likelihood for each segment, fitting
    the optimal (MLE) rate (sample mean) per segment:

    .. math::
        C(X_{s:e}) = 2 \left(n \hat{\lambda}_{s:e}(1 - \log \hat{\lambda}_{s:e})
        + \sum_{i=s}^{e-1} \log(x_i!)\right)

    where :math:`\hat{\lambda}_{s:e} = \bar{x}_{s:e}` is the MLE rate (sample mean)
    and :math:`x_i!` is the factorial of the :math:`i`-th observation.

    The cumulative sums of ``x`` and ``log(x!)`` are precomputed so that each
    segment cost is evaluated in O(1) time.

    Notes
    -----
    Requires non-negative count data. Accepts both integer and float arrays,
    provided all values are non-negative.

    Requires at least 1 observation per segment.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import PoissonCost
    >>> rng = np.random.default_rng(0)
    >>> X = rng.poisson(lam=5.0, size=(100, 2)).astype(float)
    >>> cost = PoissonCost()
    >>> cost.fit(X)
    PoissonCost()
    >>> cache = cost.precompute(X)
    >>> interval_specs = np.array([[0, 50], [50, 100]])
    >>> cost.evaluate(cache, interval_specs)
    """

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

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the cost by validating the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Must contain non-negative values.
        y : None
            Ignored.

        Returns
        -------
        self : PoissonCost
        """
        X = validate_data(self, X, ensure_2d=True, reset=True)
        if np.any(X < 0):
            raise ValueError(
                "Negative values in data passed to PoissonCost. "
                "PoissonCost requires non-negative count data."
            )
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
            Dictionary with keys:

            - ``"sums"``: cumulative column sums with a leading row of zeros,
              shape ``(n_samples + 1, n_features)``.
            - ``"log_factorial_sums"``: cumulative column sums of ``log(x!)``,
              with a leading row of zeros, shape ``(n_samples + 1, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, reset=False)
        if np.any(X < 0):
            raise ValueError(
                "Negative values in data passed to PoissonCost. "
                "PoissonCost requires non-negative count data."
            )
        from scipy.special import gammaln

        log_factorial_X = gammaln(X + 1.0)
        return {
            "sums": col_cumsum(X, init_zero=True),
            "log_factorial_sums": col_cumsum(log_factorial_X, init_zero=True),
        }

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the Poisson cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, n_features)
            Twice the negative Poisson log-likelihood for each interval and feature.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return poisson_cost(starts, ends, cache["sums"], cache["log_factorial_sums"])

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted Poisson cost.

        The Poisson model has 1 parameter per feature (the rate).

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        return bic_penalty(self.n_samples_in_, self.n_features_in_)
