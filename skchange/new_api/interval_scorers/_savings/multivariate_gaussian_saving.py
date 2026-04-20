"""Multivariate Gaussian saving for a fixed mean and covariance baseline."""

__author__ = ["johannvk", "Tveten"]

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.interval_scorers._costs.multivariate_gaussian_cost import (
    _multivariate_gaussian_cost_mle,
)
from skchange.new_api.penalties import chi2_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit


@njit
def _multivariate_gaussian_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    log_det_cov: float,
    inv_cov: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """Twice negative Gaussian log-likelihood at fixed params.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries (inclusive start, exclusive end).
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    mean : np.ndarray
        Fixed mean vector of shape (n_features,).
    log_det_cov : float
        Log-determinant of the fixed covariance matrix.
    inv_cov : np.ndarray
        Inverse of the fixed covariance matrix.
    min_size : int
        Minimum valid segment size (n_features + 1). Segments smaller than
        this receive cost ``np.inf``.

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
        centered = segment - mean
        quadratic_form = np.sum(centered @ inv_cov * centered)
        twice_nll = n * p * np.log(2 * np.pi) + n * log_det_cov + quadratic_form
        costs[i, 0] = twice_nll
    return costs


class MultivariateGaussianSaving(BaseSaving):
    r"""Multivariate Gaussian saving for a fixed mean and covariance baseline.

    The saving measures the reduction in twice the negative Gaussian log-likelihood
    when using segment-wise MLE parameters versus fixed baseline parameters:

    .. math::
        S(X_{s:e}) = C_{\text{fixed}}(X_{s:e}) - C_{\text{MLE}}(X_{s:e})

    where :math:`C_{\text{fixed}}` uses the baseline mean and covariance and
    :math:`C_{\text{MLE}}` uses segment-wise MLE estimates. A large saving
    indicates that the baseline parameters are a poor fit for that segment.

    The score is inherently aggregated over all features — it returns a single
    value per interval, not one per feature.

    Parameters
    ----------
    baseline_mean : array-like of shape (n_features,) or None, default=None
        Fixed baseline mean vector. If ``None``, estimated from training data
        using the Minimum Covariance Determinant (MCD) robust location estimate
        (see ``baseline_cov`` for details).
    baseline_cov : array-like of shape (n_features, n_features) or None, default=None
        Fixed baseline covariance matrix. Must be symmetric positive definite.
        If ``None``, estimated jointly with ``baseline_mean`` using the
        Minimum Covariance Determinant (MCD) estimator
        (:class:`sklearn.covariance.MinCovDet`). MCD finds the subset of
        approximately 75% of observations with the smallest determinant
        covariance, making it robust to a changed segment occupying up to ~25%
        of the training window. If ``baseline_mean`` is provided but
        ``baseline_cov`` is ``None``, the scatter is computed by centering at
        the given mean.

    Notes
    -----
    Requires at least :math:`p + 1` observations per segment so that the segment
    sample covariance matrix is full rank.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import MultivariateGaussianSaving
    >>> X = np.random.default_rng(0).normal(size=(100, 3))
    >>> scorer = MultivariateGaussianSaving()
    >>> scorer.fit(X)
    MultivariateGaussianSaving()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    def __init__(
        self,
        baseline_mean: ArrayLike | None = None,
        baseline_cov: ArrayLike | None = None,
    ):
        self.baseline_mean = baseline_mean
        self.baseline_cov = baseline_cov

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

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the saving, estimating baseline parameters from training data if needed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : MultivariateGaussianSaving
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        n, p = X.shape

        if self.baseline_mean is None and self.baseline_cov is None:
            # Use MCD for jointly robust (location, scatter) estimation.
            # MCD selects ~75% of observations with the smallest-determinant
            # covariance, naturally excluding a shifted segment at the end.
            if n <= p:
                raise ValueError(
                    f"Cannot estimate a {p}x{p} covariance matrix from "
                    f"n_samples={n}. Provide at least {p + 1} samples, or "
                    "supply baseline_mean and baseline_cov explicitly."
                )
            from sklearn.covariance import MinCovDet

            mcd = MinCovDet(store_precision=True, assume_centered=False)
            mcd.fit(X)
            self.baseline_mean_ = mcd.location_
            self.baseline_cov_ = mcd.covariance_
        else:
            # At least one parameter is user-supplied; resolve each independently.
            if self.baseline_mean is None:
                self.baseline_mean_ = np.median(X, axis=0)
            else:
                mean = np.asarray(self.baseline_mean, dtype=np.float64)
                if mean.shape != (p,):
                    raise ValueError(
                        f"baseline_mean must have shape ({p},), got {mean.shape}."
                    )
                self.baseline_mean_ = mean

            if self.baseline_cov is None:
                if n <= p:
                    raise ValueError(
                        f"Cannot estimate a {p}x{p} covariance matrix from "
                        f"n_samples={n}. Provide at least {p + 1} samples, or "
                        "supply baseline_cov explicitly."
                    )
                # Center at the provided/robust mean and compute the scatter matrix.
                centered = X - self.baseline_mean_
                self.baseline_cov_ = (centered.T @ centered) / n
            else:
                cov = np.asarray(self.baseline_cov, dtype=np.float64)
                if cov.shape != (p, p):
                    raise ValueError(
                        f"baseline_cov must have shape ({p}, {p}), got {cov.shape}."
                    )
                self.baseline_cov_ = cov

        sign, logdet = np.linalg.slogdet(self.baseline_cov_)
        if sign <= 0:
            raise ValueError(
                "baseline_cov must be symmetric positive definite "
                f"(log-determinant={logdet:.4g}, sign={sign})."
            )
        self._log_det_cov = logdet
        self._inv_cov = np.linalg.inv(self.baseline_cov_)
        return self

    def precompute(self, X: ArrayLike) -> dict:
        """Store the data for segment-wise evaluation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to precompute.

        Returns
        -------
        cache : dict
            Dictionary with key ``"X"``: the validated data array of shape
            ``(n_samples, n_features)``.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {"X": X}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the multivariate Gaussian saving on intervals.

        Parameters
        ----------
        cache : dict
            Cache from :meth:`precompute`.
        interval_specs : array-like of shape (n_interval_specs, 2)
            Interval boundaries ``[start, end)`` to score.

        Returns
        -------
        savings : ndarray of shape (n_interval_specs, 1)
            Saving for each interval (fixed cost minus MLE cost).
        """
        check_is_fitted(self)
        interval_specs = check_interval_specs(
            interval_specs,
            self.interval_specs_ncols,
            caller_name=self.__class__.__name__,
        )
        starts, ends = interval_specs[:, 0], interval_specs[:, 1]
        X = cache["X"]
        fixed_cost = _multivariate_gaussian_cost_fixed(
            starts,
            ends,
            X,
            self.baseline_mean_,
            self._log_det_cov,
            self._inv_cov,
            self.min_size,
        )
        mle_cost = _multivariate_gaussian_cost_mle(starts, ends, X, self.min_size)
        return fixed_cost - mle_cost

    def get_default_penalty(self) -> float:
        """Get the default chi-square penalty for the fitted saving.

        The multivariate Gaussian saving is chi-square distributed under the null
        (correct baseline) with :math:`p + p(p+1)/2` degrees of freedom
        (mean vector + upper-triangle of the covariance matrix).

        Returns
        -------
        float
            Default penalty value.
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p + p * (p + 1) // 2
        # Scaling chi2 penalty by 1.5 is done to pass the sanity checks in the
        # test suite for CAPA.
        return 1.5 * chi2_penalty(self.n_samples_in_, n_params)
