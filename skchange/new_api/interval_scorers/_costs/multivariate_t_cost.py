"""Multivariate T-distribution (twice negative log-likelihood) cost."""

__author__ = ["johannvk", "Tveten"]

from numbers import Integral, Real

import numpy as np
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.interval_scorers._base import BaseCost
from skchange.new_api.interval_scorers._costs.multivariate_gaussian_cost import (
    _multivariate_gaussian_cost_mle,
)
from skchange.new_api.penalties import bic_penalty
from skchange.new_api.typing import ArrayLike
from skchange.new_api.utils._param_validation import Interval, _fit_context
from skchange.new_api.utils._tags import SkchangeTags
from skchange.new_api.utils.validation import check_interval_specs, validate_data
from skchange.utils.numba import njit, prange
from skchange.utils.numba.stats import (
    col_median,
    digamma,
    kurtosis,
    log_gamma,
    trigamma,
)


@njit
def _estimate_scale_matrix_trace(
    centered_sample_squared_norms: np.ndarray,
    non_zero_norm_mask: np.ndarray,
    sample_dimension: int,
    dof: float,
    loo_index=-1,
):
    """Estimate the trace of the MLE multivariate T scale matrix.

    Using an isotropic estimate of the multivariate T distribution,
    estimates the trace of the MLE scale matrix from the squared norms
    of centered samples. See [1]_.

    References
    ----------
    .. [1] Aeschliman, C., Park, J., & Kak, A. C. (2009). A novel parameter
       estimation algorithm for the multivariate T-distribution. ECCV 2010, 594-607.
    """
    if loo_index >= 0:
        centered_sample_squared_norms[loo_index] = 0.0
        non_zero_norm_mask[loo_index] = False

    z_bar = np.log(centered_sample_squared_norms[non_zero_norm_mask]).mean()
    log_alpha = (
        z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(sample_dimension / 2.0)
    )
    return sample_dimension * np.exp(log_alpha)


@njit
def _initial_scale_matrix_estimate(
    centered_samples: np.ndarray,
    dof: float,
    loo_index: int = -1,
):
    """Estimate the scale matrix given centered samples and degrees of freedom.

    The direction is estimated from standardized centered samples and the
    trace from the squared norms, as described in [1]_.

    References
    ----------
    .. [1] Aeschliman, C., Park, J., & Kak, A. C. (2009). A novel parameter
       estimation algorithm for the multivariate T-distribution. ECCV 2010, 594-607.
    """
    num_samples, sample_dimension = centered_samples.shape

    centered_sample_squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    non_zero_norm_mask = centered_sample_squared_norms > 1.0e-6

    centered_sample_weights = np.ones(num_samples)
    centered_sample_weights[non_zero_norm_mask] *= (
        1.0 / centered_sample_squared_norms[non_zero_norm_mask]
    )
    weighted_samples = centered_samples * centered_sample_weights[:, np.newaxis]
    mle_scale_estimate = weighted_samples.T @ centered_samples

    if loo_index >= 0:
        loo_sample = centered_samples[loo_index, :].reshape(-1, 1)
        mle_scale_estimate -= (
            centered_sample_weights[loo_index] * loo_sample @ loo_sample.T
        )
        mle_scale_estimate /= num_samples - 1
    else:
        mle_scale_estimate /= num_samples

    scale_trace_estimate = _estimate_scale_matrix_trace(
        centered_sample_squared_norms=centered_sample_squared_norms,
        non_zero_norm_mask=non_zero_norm_mask,
        sample_dimension=sample_dimension,
        dof=dof,
        loo_index=loo_index,
    )
    mle_scale_estimate *= scale_trace_estimate / np.trace(mle_scale_estimate)

    return mle_scale_estimate


@njit
def _scale_matrix_fixed_point_iteration(
    scale_matrix: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    loo_index: int = -1,
):
    """Compute one MLE scale-matrix fixed-point iteration."""
    num_samples, sample_dim = centered_samples.shape

    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(sample_dim))
    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    sample_weights = (sample_dim + dof) / (dof + mahalanobis_squared_distances)
    weighted_samples = centered_samples * sample_weights[:, np.newaxis]

    reconstructed_scale_matrix = weighted_samples.T @ centered_samples
    if loo_index >= 0:
        loo_sample = centered_samples[loo_index, :].reshape(-1, 1)
        reconstructed_scale_matrix -= (
            sample_weights[loo_index] * loo_sample @ loo_sample.T
        )
        reconstructed_scale_matrix /= num_samples - 1
    else:
        reconstructed_scale_matrix /= num_samples

    return reconstructed_scale_matrix


@njit
def _solve_for_mle_scale_matrix(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    dof: float,
    max_iter: int,
    abs_tol: float,
    rel_tol: float,
    loo_index: int = -1,
) -> np.ndarray:
    """Run fixed-point iterations to compute the MLE scale matrix."""
    scale_matrix = initial_scale_matrix.copy()
    for iteration in range(1, max_iter + 1):
        temp_cov_matrix = _scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            dof=dof,
            centered_samples=centered_samples,
            loo_index=loo_index,
        )

        absolute_residual_norm = np.linalg.norm(temp_cov_matrix - scale_matrix)
        relative_residual_norm = absolute_residual_norm / max(
            np.linalg.norm(scale_matrix), 1.0e-12
        )

        scale_matrix[:, :] = temp_cov_matrix[:, :]
        if absolute_residual_norm < abs_tol or relative_residual_norm < rel_tol:
            break

    if iteration == max_iter:
        raise RuntimeError(
            f"MultivariateTCost: maximum iterations reached ({max_iter}) in MLE "
            "scale matrix estimation. Relax mle_scale_abs_tol / mle_scale_rel_tol "
            "or increase mle_scale_max_iter."
        )

    return scale_matrix


@njit
def maximum_likelihood_mv_t_scale_matrix(
    centered_samples: np.ndarray,
    dof: float,
    abs_tol: float,
    rel_tol: float,
    max_iter: int,
    loo_index: int = -1,
) -> np.ndarray:
    """Compute the MLE scale matrix for a multivariate T-distribution.

    Parameters
    ----------
    centered_samples : np.ndarray
        Centered samples of shape (n_samples, n_features).
    dof : float
        Degrees of freedom.
    abs_tol, rel_tol : float
        Absolute and relative convergence tolerances for fixed-point iterations.
    max_iter : int
        Maximum number of fixed-point iterations.
    loo_index : int, default=-1
        Index of the leave-one-out sample. -1 means use all samples.

    Returns
    -------
    np.ndarray
        MLE scale matrix of shape (n_features, n_features).
    """
    initial_mle_scale_matrix = _initial_scale_matrix_estimate(
        centered_samples, dof, loo_index=loo_index
    )
    return _solve_for_mle_scale_matrix(
        initial_scale_matrix=initial_mle_scale_matrix,
        centered_samples=centered_samples,
        dof=dof,
        loo_index=loo_index,
        max_iter=max_iter,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


@njit
def _multivariate_t_log_likelihood(
    scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    dof: float,
) -> float:
    """Log-likelihood of multivariate T, computing inverse and log-det internally."""
    num_samples, sample_dim = centered_samples.shape

    sign_det, log_det_scale_matrix = np.linalg.slogdet(scale_matrix)
    if sign_det <= 0:
        return np.nan
    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(sample_dim))

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


@njit
def _mv_t_ll_at_mle_params(
    X: np.ndarray,
    start: int,
    end: int,
    dof: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> float:
    """Multivariate-T log-likelihood at MLE parameters for one segment."""
    X_segment = X[start:end]
    sample_medians = col_median(X_segment)
    X_centered = X_segment - sample_medians

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        X_centered,
        dof,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
        max_iter=mle_scale_max_iter,
    )
    return _multivariate_t_log_likelihood(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, dof=dof
    )


@njit(parallel=True)
def multivariate_t_cost_mle_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    dof: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> np.ndarray:
    """Twice negative multivariate-T log-likelihood at MLE parameters.

    Parameters
    ----------
    starts, ends : np.ndarray
        Segment boundaries (inclusive start, exclusive end).
    X : np.ndarray
        Data of shape (n_samples, n_features).
    dof : float
        Degrees of freedom.
    mle_scale_abs_tol, mle_scale_rel_tol : float
        Convergence tolerances for scale-matrix fixed-point iterations.
    mle_scale_max_iter : int
        Maximum iterations for scale-matrix estimation.

    Returns
    -------
    costs : np.ndarray of shape (n_intervals, 1)
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))
    for i in prange(num_starts):
        segment_ll = _mv_t_ll_at_mle_params(
            X,
            starts[i],
            ends[i],
            dof=dof,
            mle_scale_abs_tol=mle_scale_abs_tol,
            mle_scale_rel_tol=mle_scale_rel_tol,
            mle_scale_max_iter=mle_scale_max_iter,
        )
        costs[i, 0] = -2.0 * segment_ll
    return costs


@njit
def _isotropic_mv_t_dof_estimate(
    centered_samples: np.ndarray, infinite_dof_threshold, zero_norm_tol=1.0e-6
) -> float:
    """Isotropic degrees-of-freedom estimate from log-norm variance."""
    sample_dim = centered_samples.shape[1]

    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    log_norm_sq_var = np.log(squared_norms[squared_norms > zero_norm_tol**2]).var()

    b = log_norm_sq_var - trigamma(sample_dim / 2.0)
    inf_dof_b_threshold = (2 * infinite_dof_threshold + 4) / (infinite_dof_threshold**2)
    if b < inf_dof_b_threshold:
        return np.inf

    return (1 + np.sqrt(1 + 4 * b)) / b


@njit
def _kurtosis_mv_t_dof_estimate(
    centered_samples: np.ndarray, infinite_dof_threshold: float
) -> float:
    """Kurtosis-based degrees-of-freedom estimate."""
    sample_ellipitical_kurtosis = kurtosis(centered_samples).mean() / 3.0

    inf_dof_kurtosis_threshold = 2.0 / (infinite_dof_threshold - 4.0)
    if sample_ellipitical_kurtosis < inf_dof_kurtosis_threshold:
        return np.inf

    return (2.0 / sample_ellipitical_kurtosis) + 4.0


@njit
def _iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
    dof_abs_tol=1.0e-1,
    dof_rel_tol=5.0e-2,
    dof_max_iter=10,
) -> float:
    """Estimate degrees-of-freedom iteratively. See [1]_.

    References
    ----------
    .. [1] Ollila, E., Palomar, D. P., & Pascal, F. (2020). Shrinking the eigenvalues
       of M-estimators of covariance matrix. IEEE Trans. Signal Process., 256-269.
    """
    n = centered_samples.shape[0]
    if initial_dof > infinite_dof_threshold:
        return np.inf

    inf_dof_nu_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)
    sample_covariance = (centered_samples.T @ centered_samples) / n

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        initial_dof,
        max_iter=mle_scale_max_iter,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
    )

    dof = initial_dof
    for _ in range(dof_max_iter):
        nu_i = np.trace(sample_covariance) / np.trace(mle_scale_matrix)
        if nu_i < inf_dof_nu_threshold:
            dof = np.inf
            break

        old_dof = dof
        dof = 2 * nu_i / max((nu_i - 1), 1.0e-3)

        mle_scale_matrix = _solve_for_mle_scale_matrix(
            initial_scale_matrix=mle_scale_matrix,
            centered_samples=centered_samples,
            dof=dof,
            abs_tol=mle_scale_abs_tol,
            rel_tol=mle_scale_rel_tol,
            max_iter=mle_scale_max_iter,
        )

        absolute_dof_diff = np.abs(dof - old_dof)
        if absolute_dof_diff / old_dof < dof_rel_tol or absolute_dof_diff < dof_abs_tol:
            break

    return dof


@njit(parallel=True)
def _loo_iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
    dof_abs_tol=1.0e-1,
    dof_rel_tol=5.0e-2,
    dof_max_iter=5,
) -> float:
    """Leave-one-out iterative degrees-of-freedom estimate. See [1]_.

    References
    ----------
    .. [1] Pascal, F., Ollila, E., & Palomar, D. P. (2021). Improved estimation of
       the degree of freedom parameter of multivariate T-distribution. EUSIPCO, 860-864.
    """
    if initial_dof > infinite_dof_threshold:
        return np.inf

    num_samples, sample_dimension = centered_samples.shape
    inf_dof_theta_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)

    sample_covariance = (centered_samples.T @ centered_samples) / num_samples
    grand_mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        dof=initial_dof,
        max_iter=mle_scale_max_iter,
        abs_tol=mle_scale_abs_tol,
        rel_tol=mle_scale_rel_tol,
    )
    contraction_estimate = np.trace(grand_mle_scale_matrix) / np.trace(
        sample_covariance
    )

    current_dof = initial_dof

    for _ in range(dof_max_iter):
        total_loo_mahalanobis_squared_distance = 0.0
        for sample in prange(num_samples):
            loo_sample = centered_samples[sample, :].reshape(-1, 1)
            loo_sample_outer_product = loo_sample @ loo_sample.T

            loo_scale_estimate = grand_mle_scale_matrix - contraction_estimate * (
                loo_sample_outer_product / num_samples
            )

            loo_mle_scale_matrix = _solve_for_mle_scale_matrix(
                initial_scale_matrix=loo_scale_estimate,
                centered_samples=centered_samples,
                dof=current_dof,
                loo_index=sample,
                abs_tol=mle_scale_abs_tol,
                rel_tol=mle_scale_rel_tol,
                max_iter=mle_scale_max_iter,
            )

            loo_mahalanobis_squared_distance = (
                loo_sample.T @ np.linalg.solve(loo_mle_scale_matrix, loo_sample)
            )[0, 0]
            total_loo_mahalanobis_squared_distance += loo_mahalanobis_squared_distance

        theta_k = (1 - sample_dimension / num_samples) * (
            (total_loo_mahalanobis_squared_distance / num_samples) / sample_dimension
        )
        if theta_k < inf_dof_theta_threshold:
            current_dof = np.inf
            break

        new_dof = 2 * theta_k / (theta_k - 1)
        abs_dof_difference = np.abs(new_dof - current_dof)
        if (
            abs_dof_difference < dof_abs_tol
            or (abs_dof_difference / current_dof) < dof_rel_tol
        ):
            current_dof = new_dof
            break

        current_dof = new_dof

    return current_dof


@njit
def _estimate_mv_t_dof(
    X: np.ndarray,
    infinite_dof_threshold: float,
    refine_dof_threshold: int,
    mle_scale_abs_tol: float,
    mle_scale_rel_tol: float,
    mle_scale_max_iter: int,
) -> float:
    """Estimate the degrees of freedom of a multivariate T-distribution.

    Combines an isotropic estimator [1]_, a kurtosis estimator, and an iterative
    refinement [2]_. Optionally applies a LOO refinement [3]_ when the sample
    count is small.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Data matrix.
    infinite_dof_threshold : float
        Above this value the distribution is treated as Gaussian.
    refine_dof_threshold : int
        If n_samples <= this, apply the LOO refinement.
    mle_scale_abs_tol, mle_scale_rel_tol : float
        Convergence tolerances for scale-matrix estimation.
    mle_scale_max_iter : int
        Maximum scale-matrix iterations.

    Returns
    -------
    float
        Estimated degrees of freedom.

    References
    ----------
    .. [1] Aeschliman, C., Park, J., & Kak, A. C. (2009). ECCV 2010, 594-607.
    .. [2] Ollila, E., Palomar, D. P., & Pascal, F. (2020). IEEE Trans. Signal
       Process., 256-269.
    .. [3] Pascal, F., Ollila, E., & Palomar, D. P. (2021). EUSIPCO, 860-864.
    """
    centered_samples = X - col_median(X)

    isotropic_dof = _isotropic_mv_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )
    kurtosis_dof = _kurtosis_mv_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )

    if np.isfinite(isotropic_dof) and np.isfinite(kurtosis_dof):
        initial_dof_estimate = np.sqrt(isotropic_dof * kurtosis_dof)
    elif np.isfinite(isotropic_dof):
        initial_dof_estimate = isotropic_dof
    elif np.isfinite(kurtosis_dof):
        initial_dof_estimate = kurtosis_dof
    else:
        initial_dof_estimate = infinite_dof_threshold / 2.0

    dof_estimate = _iterative_mv_t_dof_estimate(
        centered_samples=centered_samples,
        initial_dof=initial_dof_estimate,
        infinite_dof_threshold=infinite_dof_threshold,
        mle_scale_abs_tol=mle_scale_abs_tol,
        mle_scale_rel_tol=mle_scale_rel_tol,
        mle_scale_max_iter=mle_scale_max_iter,
    )

    if X.shape[0] <= refine_dof_threshold:
        dof_estimate = _loo_iterative_mv_t_dof_estimate(
            centered_samples=centered_samples,
            initial_dof=dof_estimate,
            infinite_dof_threshold=infinite_dof_threshold,
            mle_scale_abs_tol=mle_scale_abs_tol,
            mle_scale_rel_tol=mle_scale_rel_tol,
            mle_scale_max_iter=mle_scale_max_iter,
        )

    return dof_estimate


class MultivariateTCost(BaseCost):
    r"""Multivariate T-distribution twice negative log-likelihood cost.

    Computes twice the negative multivariate-T log-likelihood for each segment,
    fitting the MLE mean (componentwise median) and scale matrix per segment:

    .. math::
        C(X_{s:e}) = -2 \sum_{i=s}^{e-1} \log p(x_i \mid \hat{\mu}, \hat{\Sigma},
        \hat{\nu})

    where :math:`\hat{\mu}` is the segment median, :math:`\hat{\Sigma}` is the
    MLE scale matrix (found by fixed-point iteration), and :math:`\hat{\nu}` is
    the degrees of freedom estimated from the training data (or fixed by the user).

    The multivariate-T is heavier-tailed than the Gaussian, making this cost
    more robust to outliers. When the estimated degrees of freedom exceed
    ``infinite_dof_threshold``, the cost falls back to the Gaussian MLE cost.

    The score is inherently aggregated over all features — it returns a single
    value per interval, not one per feature.

    Parameters
    ----------
    fixed_dof : float or None, default=None
        Fixed degrees of freedom. If ``None``, estimated from the training data
        using a combination of an isotropic estimator, a kurtosis-based estimator,
        and an iterative refinement procedure. Estimation is the most expensive
        part of fitting.
    infinite_dof_threshold : float, default=50.0
        When the estimated (or fixed) degrees of freedom exceed this value, the
        cost falls back to the Gaussian MLE cost.
    refine_dof_threshold : int or None, default=None
        Number of training samples below which a leave-one-out iterative dof
        refinement is applied. Defaults to 1000 when Numba is available and 100
        otherwise.
    mle_scale_abs_tol : float, default=1e-2
        Absolute convergence tolerance for the MLE scale-matrix fixed-point
        iterations.
    mle_scale_rel_tol : float, default=1e-2
        Relative convergence tolerance for the MLE scale-matrix fixed-point
        iterations.
    mle_scale_max_iter : int, default=100
        Maximum number of fixed-point iterations for scale-matrix estimation.
        Raises ``RuntimeError`` if reached.

    Notes
    -----
    Requires at least :math:`p + 1` observations per segment so that the scale
    matrix estimate is full rank.

    Between 5x and 10x slower than :class:`MultivariateGaussianCost` when Numba
    is installed, due to the per-segment fixed-point iterations.

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
    >>> from skchange.new_api.interval_scorers import MultivariateTCost
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_t(df=5, size=(100, 3))
    >>> cost = MultivariateTCost()
    >>> cost.fit(X)
    MultivariateTCost()
    >>> cache = cost.precompute(X)
    >>> cost.evaluate(cache, np.array([[0, 50], [50, 100]]))
    """

    _parameter_constraints: dict = {
        "fixed_dof": [Interval(Real, 0, None, closed="neither"), None],
        "infinite_dof_threshold": [Interval(Real, 0, None, closed="neither")],
        "refine_dof_threshold": [Interval(Integral, 1, None, closed="left"), None],
        "mle_scale_abs_tol": [Interval(Real, 0, None, closed="neither")],
        "mle_scale_rel_tol": [Interval(Real, 0, None, closed="neither")],
        "mle_scale_max_iter": [Interval(Integral, 1, None, closed="left")],
    }

    def __init__(
        self,
        fixed_dof: float | None = None,
        infinite_dof_threshold: float = 50.0,
        refine_dof_threshold: int | None = None,
        mle_scale_abs_tol: float = 1e-2,
        mle_scale_rel_tol: float = 1e-2,
        mle_scale_max_iter: int = 100,
    ):
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
        """Fit the cost by estimating degrees of freedom from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        self : MultivariateTCost
        """
        X = validate_data(self, X, ensure_2d=True, dtype=np.float64, reset=True)
        n, p = X.shape

        if n <= p and self.fixed_dof is None:
            raise ValueError(
                f"Cannot estimate degrees of freedom from n_samples={n} with "
                f"n_features={p}. Provide at least {p + 1} samples, or set "
                "fixed_dof explicitly."
            )

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
        """Evaluate the multivariate-T cost on intervals.

        Parameters
        ----------
        cache : dict
            Output from :meth:`precompute`.
        interval_specs : array-like of shape (n_intervals, 2)
            Each row is ``[start, end)`` defining a segment.

        Returns
        -------
        costs : ndarray of shape (n_intervals, 1)
            Twice negative log-likelihood for each interval.
        """
        X = cache["X"]
        interval_specs = check_interval_specs(interval_specs, self.interval_specs_ncols)
        starts = interval_specs[:, 0]
        ends = interval_specs[:, 1]

        if np.isposinf(self.dof_):
            return _multivariate_gaussian_cost_mle(starts, ends, X, self.min_size)
        else:
            return multivariate_t_cost_mle_params(
                starts,
                ends,
                X=X,
                dof=self.dof_,
                mle_scale_abs_tol=self.mle_scale_abs_tol,
                mle_scale_rel_tol=self.mle_scale_rel_tol,
                mle_scale_max_iter=self.mle_scale_max_iter,
            )

    def get_default_penalty(self) -> float:
        """Get the default BIC penalty for the fitted cost.

        The model has :math:`p + p(p+1)/2 + 1` free parameters (mean, scale
        matrix upper triangle, dof), but dof is estimated once from training
        data rather than per segment, so the effective per-segment count is
        :math:`p + p(p+1)/2`.

        Returns
        -------
        float
            BIC penalty value.
        """
        check_is_fitted(self)
        p = self.n_features_in_
        n_params = p + p * (p + 1) // 2
        return bic_penalty(self.n_samples_in_, n_params)
