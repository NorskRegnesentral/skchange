"""Multivariate T-distribution saving for a fixed mean and scale baseline."""

__author__ = ["johannvk", "Tveten"]

from numbers import Integral, Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseSaving
from skchange.new_api.interval_scorers._costs.multivariate_gaussian_cost import (
    _multivariate_gaussian_cost_mle,
)
from skchange.new_api.interval_scorers._costs.multivariate_t_cost import (
    _estimate_mv_t_dof,
    multivariate_t_cost_mle_params,
)
from skchange.new_api.interval_scorers._savings.multivariate_gaussian_saving import (
    _multivariate_gaussian_cost_fixed,
)
from skchange.new_api.penalties import chi2_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import Interval, _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit, prange
from skchange.utils.numba.stats import log_gamma


@njit
def _fixed_param_multivariate_t_log_likelihood(
    centered_samples: np.ndarray,
    dof: float,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
) -> float:
    """Log-likelihood of multivariate T at fixed (pre-computed) parameters."""
    num_samples, sample_dim = centered_samples.shape

    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    exponent = 0.5 * (dof + sample_dim)
    A = log_gamma(exponent)
    B = log_gamma(0.5 * dof)
    C = 0.5 * sample_dim * np.log(dof * np.pi)
    D = 0.5 * log_det_scale_matrix

    normalization_contribution = num_samples * (A - B - C - D)
    sample_contributions = -exponent * np.log1p(mahalanobis_squared_distances / dof)
    return normalization_contribution + sample_contributions.sum()


@njit(parallel=True)
def _multivariate_t_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
    dof: float,
    min_size: int,
) -> np.ndarray:
    """Twice negative multivariate-T log-likelihood at fixed parameters.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries (inclusive start, exclusive end).
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    mean : np.ndarray
        Fixed mean vector of shape (n_features,).
    inverse_scale_matrix : np.ndarray
        Inverse of the fixed scale matrix, shape (n_features, n_features).
    log_det_scale_matrix : float
        Log-determinant of the fixed scale matrix.
    dof : float
        Degrees of freedom.
    min_size : int
        Minimum valid segment size. Segments smaller than this receive np.inf.

    Returns
    -------
    costs : np.ndarray of shape (n_intervals, 1)
    """
    n_intervals = len(starts)
    costs = np.empty((n_intervals, 1))
    for i in prange(n_intervals):
        n = ends[i] - starts[i]
        if n < min_size:
            costs[i, 0] = np.inf
            continue
        centered = X[starts[i] : ends[i]] - mean
        ll = _fixed_param_multivariate_t_log_likelihood(
            centered_samples=centered,
            dof=dof,
            inverse_scale_matrix=inverse_scale_matrix,
            log_det_scale_matrix=log_det_scale_matrix,
        )
        costs[i, 0] = -2.0 * ll
    return costs


class MultivariateTSaving(BaseSaving):
    r"""Multivariate T-distribution saving for a fixed mean and scale baseline.

    The saving measures the reduction in twice the negative multivariate-T
    log-likelihood when using segment-wise MLE parameters versus fixed baseline
    parameters:

    .. math::
        S(X_{s:e}) = C_{\text{fixed}}(X_{s:e}) - C_{\text{MLE}}(X_{s:e})

    where :math:`C_{\text{fixed}}` uses the baseline mean, scale matrix and dof
    and :math:`C_{\text{MLE}}` uses segment-wise MLE estimates. A large saving
    indicates that the baseline parameters are a poor fit for that segment.

    The score is inherently aggregated over all features — it returns a single
    value per interval, not one per feature.

    Parameters
    ----------
    baseline_mean : array-like of shape (n_features,) or None, default=None
        Fixed baseline mean vector. If ``None``, estimated from training data
        using a median-based trimmed estimator.
    baseline_scale : array-like of shape (n_features, n_features) or None, default=None
        Fixed baseline scale matrix. Must be symmetric positive definite. If
        ``None``, estimated jointly with ``baseline_mean`` by discarding the 25%
        of observations furthest from the componentwise median (measured in
        MAD-scaled distance) and computing the sample covariance of the remaining
        75%. This approach is robust to multi-modal contamination in the training
        window.
    fixed_dof : float or None, default=None
        Fixed degrees of freedom for both baseline and MLE evaluation. If ``None``,
        estimated from the training data.
    infinite_dof_threshold : float, default=50.0
        When dof exceeds this value, the cost falls back to the Gaussian likelihood.
    refine_dof_threshold : int or None, default=None
        Training-sample count below which LOO iterative dof refinement is applied.
        Defaults to 1000 (Numba) or 100 (pure Python).
    mle_scale_abs_tol : float, default=1e-2
        Absolute convergence tolerance for scale-matrix fixed-point iterations.
    mle_scale_rel_tol : float, default=1e-2
        Relative convergence tolerance for scale-matrix fixed-point iterations.
    mle_scale_max_iter : int, default=100
        Maximum fixed-point iterations. Raises ``RuntimeError`` if reached.

    Notes
    -----
    Requires at least :math:`p + 1` observations per segment so that the MLE
    scale matrix is full rank.

    References
    ----------
    .. [1] Aeschliman, C., Park, J., & Kak, A. C. (2009). A novel parameter
       estimation algorithm for the multivariate T-distribution. ECCV 2010, 594-607.
    .. [2] Ollila, E., Palomar, D. P., & Pascal, F. (2020). Shrinking the eigenvalues
       of M-estimators of covariance matrix. IEEE Trans. Signal Process., 256-269.
    .. [3] Pascal, F., Ollila, E., & Palomar, D. P. (2021). Improved estimation of
       the degree of freedom parameter of multivariate T-distribution. EUSIPCO, 860-864.

    Examples
    --------
    >>> import numpy as np
    >>> from skchange.new_api.interval_scorers import MultivariateTSaving
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_t(df=5, size=(100, 3))
    >>> scorer = MultivariateTSaving()
    >>> scorer.fit(X)
    MultivariateTSaving()
    >>> cache = scorer.precompute(X)
    >>> scorer.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    _parameter_constraints: dict = {
        "baseline_mean": ["array-like", None],
        "baseline_scale": ["array-like", None],
        "fixed_dof": [Interval(Real, 0, None, closed="neither"), None],
        "infinite_dof_threshold": [Interval(Real, 0, None, closed="neither")],
        "refine_dof_threshold": [Interval(Integral, 1, None, closed="left"), None],
        "mle_scale_abs_tol": [Interval(Real, 0, None, closed="neither")],
        "mle_scale_rel_tol": [Interval(Real, 0, None, closed="neither")],
        "mle_scale_max_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        baseline_mean: ArrayLike | None = None,
        baseline_scale: ArrayLike | None = None,
        fixed_dof: float | None = None,
        infinite_dof_threshold: float = 50.0,
        refine_dof_threshold: int | None = None,
        mle_scale_abs_tol: float = 1e-2,
        mle_scale_rel_tol: float = 1e-2,
        mle_scale_max_iter: int = 100,
    ):
        self.baseline_mean = baseline_mean
        self.baseline_scale = baseline_scale
        self.fixed_dof = fixed_dof
        self.infinite_dof_threshold = infinite_dof_threshold
        self.refine_dof_threshold = refine_dof_threshold
        self.mle_scale_abs_tol = mle_scale_abs_tol
        self.mle_scale_rel_tol = mle_scale_rel_tol
        self.mle_scale_max_iter = mle_scale_max_iter

    def __sklearn_tags__(self) -> SkchangeTags:
        """Return tags marking this scorer as aggregated."""
        tags = super().__sklearn_tags__()
        tags.interval_scorer_tags.aggregated = True
        return tags

    @property
    def min_size(self) -> int:
        """Minimum segment size (n_features + 1, for a full-rank scale matrix)."""
        check_is_fitted(self)
        return self.n_features_in_ + 1

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Fit the saving, estimating baseline parameters and dof from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : MultivariateTSaving
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        n, p = X.shape

        if n <= p:
            raise ValueError(
                f"Cannot estimate a {p}x{p} scale matrix from n_samples={n}. "
                f"Provide at least {p + 1} samples, or supply baseline parameters."
            )

        # --- Estimate degrees of freedom ---
        if self.fixed_dof is None:
            refine_threshold = self.refine_dof_threshold
            if refine_threshold is None:
                from skchange.utils.numba import numba_available

                refine_threshold = 1000 if numba_available else 100
            self.dof_ = float(
                _estimate_mv_t_dof(
                    X,
                    infinite_dof_threshold=self.infinite_dof_threshold,
                    refine_dof_threshold=refine_threshold,
                    mle_scale_abs_tol=self.mle_scale_abs_tol,
                    mle_scale_rel_tol=self.mle_scale_rel_tol,
                    mle_scale_max_iter=self.mle_scale_max_iter,
                )
            )
        else:
            self.dof_ = float(self.fixed_dof)

        # --- Estimate baseline mean and scale matrix ---
        if self.baseline_mean is None and self.baseline_scale is None:
            if n <= p:
                raise ValueError(
                    f"Cannot estimate a {p}x{p} scale matrix from "
                    f"n_samples={n}. Provide at least {p + 1} samples, or "
                    "supply baseline_mean and baseline_scale explicitly."
                )
            from sklearn.covariance import MinCovDet

            mcd = MinCovDet(store_precision=True, assume_centered=False)
            mcd.fit(X)
            self.baseline_mean_ = mcd.location_
            self.baseline_scale_ = mcd.covariance_
        else:
            if self.baseline_mean is None:
                self.baseline_mean_ = np.median(X, axis=0)
            else:
                mean = np.asarray(self.baseline_mean, dtype=np.float64)
                if mean.shape != (p,):
                    raise ValueError(
                        f"baseline_mean must have shape ({p},), got {mean.shape}."
                    )
                self.baseline_mean_ = mean

            if self.baseline_scale is None:
                centered = X - self.baseline_mean_
                self.baseline_scale_ = np.atleast_2d(np.cov(centered.T))
            else:
                scale = np.asarray(self.baseline_scale, dtype=np.float64)
                if scale.shape != (p, p):
                    raise ValueError(
                        f"baseline_scale must have shape ({p}, {p}), got {scale.shape}."
                    )
                self.baseline_scale_ = scale

        sign, logdet = np.linalg.slogdet(self.baseline_scale_)
        if sign <= 0:
            raise ValueError(
                "baseline_scale must be symmetric positive definite "
                f"(log-determinant={logdet:.4g}, sign={sign})."
            )
        self._log_det_scale = logdet
        self._inv_scale = np.linalg.inv(self.baseline_scale_)
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
            Dictionary with key ``"X"``: the validated data array.
        """
        check_is_fitted(self)
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=False)
        return {"X": X}

    def evaluate(self, cache: dict, interval_specs: ArrayLike) -> np.ndarray:
        """Evaluate the multivariate-T saving on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 2)
            Each row is ``[start, end)`` defining a segment.

        Returns
        -------
        savings : ndarray of shape (n_intervals, 1)
            Saving (fixed-param cost minus MLE cost) for each interval.
        """
        X = cache["X"]
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        ends = interval_specs[:, 1]
        min_size = self.min_size

        if np.isposinf(self.dof_):
            cost_fixed = _multivariate_gaussian_cost_fixed(
                starts,
                ends,
                X,
                mean=self.baseline_mean_,
                log_det_cov=self._log_det_scale,
                inv_cov=self._inv_scale,
                min_size=min_size,
            )
            cost_mle = _multivariate_gaussian_cost_mle(starts, ends, X, min_size)
        else:
            cost_fixed = _multivariate_t_cost_fixed_params(
                starts,
                ends,
                X,
                mean=self.baseline_mean_,
                inverse_scale_matrix=self._inv_scale,
                log_det_scale_matrix=self._log_det_scale,
                dof=self.dof_,
                min_size=min_size,
            )
            cost_mle = multivariate_t_cost_mle_params(
                starts,
                ends,
                X=X,
                dof=self.dof_,
                mle_scale_abs_tol=self.mle_scale_abs_tol,
                mle_scale_rel_tol=self.mle_scale_rel_tol,
                mle_scale_max_iter=self.mle_scale_max_iter,
            )

        return cost_fixed - cost_mle

    def get_default_penalty(self) -> np.ndarray:
        """Get the default penalty for the fitted saving.

        Uses the number of free parameters per feature implied by the
        multivariate-T model: mean (:math:`p`) and upper-triangular scale
        (:math:`p(p+1)/2`).

        Returns
        -------
        np.ndarray of shape (n_features,)
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p + p * (p + 1) // 2
        return 1.5 * chi2_penalty(self.n_samples_in_, n_params)
