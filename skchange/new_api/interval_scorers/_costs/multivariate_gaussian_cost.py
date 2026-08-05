"""Multivariate Gaussian (negative log-likelihood) cost."""

__author__ = ["johannvk", "Tveten"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.types import ArrayLike
from skchange.new_api.utils._numba import njit
from skchange.new_api.utils._numeric import log_det_covariance
from skchange.new_api.utils._param_validation import _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data

_MAX_N_SAMPLES_DEFAULT_CACHE = 10_000
_MAX_N_FEATURES_DEFAULT_CACHE = 100


def _multivariate_gaussian_precompute(
    X: np.ndarray,
    store_cov: bool | None,
) -> dict:
    """Build the shared cache for multivariate Gaussian interval scorers."""
    n_samples, n_features = X.shape
    if store_cov is None:
        store_cov = (
            n_samples <= _MAX_N_SAMPLES_DEFAULT_CACHE
            and n_features <= _MAX_N_FEATURES_DEFAULT_CACHE
        )

    if not store_cov:
        return {"X": X, "store_cov": False}

    feature_sums = np.zeros((n_samples + 1, n_features))
    np.cumsum(X, axis=0, out=feature_sums[1:])
    outer_product_sums = np.zeros((n_samples + 1, n_features, n_features))
    np.einsum("ni,nj->nij", X, X, out=outer_product_sums[1:])
    np.cumsum(outer_product_sums[1:], axis=0, out=outer_product_sums[1:])
    cache = {
        "feature_sums": feature_sums,
        "outer_product_sums": outer_product_sums,
        "store_cov": True,
    }
    return cache


@njit(cache=True)
def _multivariate_gaussian_cost_mle(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Twice negative Gaussian log-likelihood at MLE params.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries (inclusive start, exclusive end).
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    min_size : int
        Minimum valid segment size (n_features + 1). Segments smaller than
        this, or with a singular covariance, receive cost ``np.inf``.

    Returns
    -------
    costs : np.ndarray of shape (n_intervals, 1)
        Returns twice the negative Gaussian log-likelihood of each segment in `X`,
        and ``np.inf`` for segments smaller than `min_size`.
    """
    n_intervals = len(starts)
    p = X.shape[1]
    costs = np.empty((n_intervals, 1))
    for i in range(n_intervals):
        n = ends[i] - starts[i]
        if n < min_size:
            costs[i, 0] = np.inf
            continue
        segment = X[starts[i] : ends[i]]
        log_det_cov = log_det_covariance(segment)
        if np.isnan(log_det_cov):
            costs[i, 0] = np.inf
        else:
            twice_nll = n * p * np.log(2 * np.pi) + n * log_det_cov + p * n
            costs[i, 0] = twice_nll
    return costs


@njit(cache=True)
def _multivariate_gaussian_cost_mle_cached(
    starts: np.ndarray,
    ends: np.ndarray,
    feature_sums: np.ndarray,
    outer_product_sums: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Twice negative Gaussian log-likelihood from cumulative moments."""
    n_intervals = len(starts)
    p = feature_sums.shape[1]
    costs = np.empty((n_intervals, 1))
    for i in range(n_intervals):
        n = ends[i] - starts[i]
        if n < min_size:
            costs[i, 0] = np.inf
            continue

        partial_sum = feature_sums[ends[i]] - feature_sums[starts[i]]
        mean = partial_sum / n
        partial_outer_product_sum = (
            outer_product_sums[ends[i]] - outer_product_sums[starts[i]]
        )
        covariance = partial_outer_product_sum / n - np.outer(mean, mean)
        det_sign, log_det_cov = np.linalg.slogdet(covariance)
        if det_sign <= 0:
            costs[i, 0] = np.inf
        else:
            twice_nll = n * p * np.log(2 * np.pi) + n * log_det_cov + p * n
            costs[i, 0] = twice_nll
    return costs


def _multivariate_gaussian_cost_mle_from_cache(
    starts: np.ndarray,
    ends: np.ndarray,
    cache: dict,
    min_size: int,
) -> np.ndarray:
    """Evaluate the MLE cost using the representation stored in ``cache``."""
    if cache["store_cov"]:
        return _multivariate_gaussian_cost_mle_cached(
            starts,
            ends,
            cache["feature_sums"],
            cache["outer_product_sums"],
            min_size,
        )
    return _multivariate_gaussian_cost_mle(starts, ends, cache["X"], min_size)


class MultivariateGaussianCost(BaseCost):
    r"""Multivariate Gaussian (negative log-likelihood) cost.

    Computes twice the negative Gaussian log-likelihood for each segment, fitting
    the optimal (MLE) mean vector and covariance matrix per segment:

    .. math::
        C(X_{s:e}) = n \log\det(\hat{\Sigma}_{s:e}) + np

    where :math:`\hat{\Sigma}_{s:e}` is the MLE covariance of the segment and
    :math:`p` is the number of features.

    The score is inherently aggregated over all features — it returns a single
    value per interval, not one per feature.

    Parameters
    ----------
    store_cov : bool or None, default=None
        Whether to cache cumulative sums and cumulative outer-product sums.
        If ``None``, caching is used when the precomputed data has at most
        10,000 samples and at most 100 features. Caching uses
        :math:`O(n p^2)` memory but makes covariance calculation independent
        of interval length.

    Notes
    -----
    Requires at least :math:`p + 1` observations per segment so that the
    sample covariance matrix is full rank.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import MultivariateGaussianCost
    >>> X = np.random.default_rng(0).normal(size=(100, 3))
    >>> cost = MultivariateGaussianCost()
    >>> cost.fit(X)
    MultivariateGaussianCost()
    >>> cache = cost.precompute(X)
    >>> cost.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    _parameter_constraints: dict = {"store_cov": ["boolean", None]}

    def __init__(self, store_cov: bool | None = None):
        self.store_cov = store_cov

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as aggregated."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum segment size (n_features + 1, for a full-rank sample covariance)."""
        check_is_fitted(self)
        return self.n_features_in_ + 1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the cost by recording the number of features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : MultivariateGaussianCost
        """
        validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Precompute data for segment-wise covariance evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            If caching is enabled, contains zero-prefixed cumulative sums and
            cumulative outer-product sums. Otherwise, contains the validated
            data under ``"X"``. The resolved strategy is stored under
            ``"store_cov"``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return _multivariate_gaussian_precompute(X, self.store_cov)

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the multivariate Gaussian cost on intervals.

        Parameters
        ----------
        cache : dict
            Cache from :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        costs : ndarray of shape (n_interval_specs, 1)
            Twice the negative log-likelihood for each interval.
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        return _multivariate_gaussian_cost_mle_from_cache(
            starts, ends, cache, self.min_size
        )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted cost.

        The multivariate Gaussian model has :math:`p + p(p+1)/2` parameters
        (mean vector + upper-triangle of the covariance matrix).

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p + p * (p + 1) // 2
        return bic_penalty(self.n_samples_in_, n_params)
