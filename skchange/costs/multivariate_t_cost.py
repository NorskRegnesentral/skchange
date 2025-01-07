"""Multivariate T distribution likelihood cost."""

__author__ = ["johannvk"]
__all__ = ["MultivariateTCost"]

from typing import Union

import numpy as np

from skchange.costs.base import BaseCost
from skchange.costs.multivariate_gaussian_cost import (
    gaussian_cost_fixed_params,
    gaussian_cost_mle_params,
)
from skchange.costs.utils import CovType, MeanType, check_cov, check_mean
from skchange.utils.numba import njit, prange
from skchange.utils.numba.stats import (
    col_median,
    digamma,
    kurtosis,
    log_gamma,
    trigamma,
)


@njit
def estimate_scale_matrix_trace(centered_samples: np.ndarray, dof: float):
    """Estimate the scale parameter of the MLE covariance matrix.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    p = centered_samples.shape[1]
    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    z_bar = np.log(squared_norms[squared_norms > 1.0e-12]).mean()
    log_alpha = z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(p / 2.0)
    return p * np.exp(log_alpha)


@njit
def initial_scale_matrix_estimate(
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    apply_trace_correction: bool = True,
):
    """Estimate the scale matrix given centered samples and degrees of freedom."""
    n, p = centered_samples.shape
    num_effective_samples = n - num_zeroed_samples

    sample_covariance_matrix = (
        centered_samples.T @ centered_samples
    ) / num_effective_samples

    if apply_trace_correction:
        scale_trace_estimate = estimate_scale_matrix_trace(centered_samples, t_dof)
        sample_covariance_matrix *= scale_trace_estimate / np.trace(
            sample_covariance_matrix
        )

    return sample_covariance_matrix


@njit
def scale_matrix_fixed_point_iteration(
    scale_matrix: np.ndarray,
    dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
):
    """Compute a multivariate T MLE scale matrix fixed point iteration."""
    n, p = centered_samples.shape

    # Subtract the number of 'zeroed' samples:
    effective_num_samples = n - num_zeroed_samples

    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(p))
    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    sample_weights = (p + dof) / (dof + mahalanobis_squared_distances)
    weighted_samples = centered_samples * sample_weights[:, np.newaxis]

    reconstructed_scale_matrix = (
        weighted_samples.T @ centered_samples
    ) / effective_num_samples

    return reconstructed_scale_matrix


@njit
def solve_for_mle_scale_matrix(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    dof: float,
    num_zeroed_samples: int = 0,
    max_iter: int = 50,
    abs_tol: float = 1.0e-3,
) -> np.ndarray:
    """Perform fixed point iterations to compute the MLE scale matrix."""
    scale_matrix = initial_scale_matrix.copy()
    for iteration in range(max_iter):
        temp_cov_matrix = scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            dof=dof,
            centered_samples=centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )

        # Note: 'ord = None' computes the Frobenius norm.
        residual = np.linalg.norm(temp_cov_matrix - scale_matrix, ord=None)

        scale_matrix[:, :] = temp_cov_matrix[:, :]
        if residual < abs_tol:
            break

    return scale_matrix, iteration + 1


@njit
def maximum_likelihood_mv_t_scale_matrix(
    centered_samples: np.ndarray,
    dof: float,
    abs_tol: float = 1.0e-3,
    max_iter: int = 50,
    num_zeroed_samples: int = 0,
    initial_trace_correction: bool = True,
) -> np.ndarray:
    """Compute the MLE scale matrix for a multivariate t-distribution.

    Parameters
    ----------
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    np.ndarray
        The MLE covariance matrix of the multivariate t-distribution.
    """
    # Initialize the scale matrix maximum likelihood estimate:
    mle_scale_matrix = initial_scale_matrix_estimate(
        centered_samples,
        dof,
        num_zeroed_samples=num_zeroed_samples,
        apply_trace_correction=initial_trace_correction,
    )

    mle_scale_matrix, inner_iterations = solve_for_mle_scale_matrix(
        initial_scale_matrix=mle_scale_matrix,
        centered_samples=centered_samples,
        dof=dof,
        num_zeroed_samples=num_zeroed_samples,
        max_iter=max_iter,
        abs_tol=abs_tol,
    )
    # print(f"Inner scale matrix MLE iterations: {inner_iterations}")

    return mle_scale_matrix


@njit
def _multivariate_t_log_likelihood(
    centered_samples: np.ndarray,
    dof: float,
    scale_matrix: Union[np.ndarray, None] = None,
    inverse_scale_matrix: Union[np.ndarray, None] = None,
    log_det_scale_matrix: Union[float, None] = None,
) -> float:
    """Calculate the log likelihood of a multivariate t-distribution.

    Directly from the definition of the multivariate t-distribution.
    Implementation inspired by the scipy implementation of
    the multivariate t-distribution, but simplified.
    Source: https://en.wikipedia.org/wiki/Multivariate_t-distribution

    Parameters
    ----------
    scale_matrix : np.ndarray
        The scale matrix of the multivariate t-distribution.
    centered_samples : np.ndarray
        The centered samples from the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    float
        The log likelihood of the multivariate t-distribution.
    """
    if scale_matrix is None and (
        inverse_scale_matrix is None or log_det_scale_matrix is None
    ):
        raise ValueError(
            "Either the scale matrix by itself, or the inverse scale matrix "
            "and log determinant of the scale matrix must be provided."
        )
    elif (log_det_scale_matrix is None) ^ (inverse_scale_matrix is None):
        raise ValueError(
            "Both the log determinant of the scale matrix and the inverse "
            "scale matrix must be provided to compute the log likelihood."
        )

    num_samples, sample_dim = centered_samples.shape

    if log_det_scale_matrix is None and inverse_scale_matrix is None:
        sign_det, log_det_scale_matrix = np.linalg.slogdet(scale_matrix)
        if sign_det <= 0:
            # TODO, raise a warning here?
            return np.nan

        inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(sample_dim))

    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    # Normalization constants:
    exponent = 0.5 * (dof + sample_dim)
    A = log_gamma(exponent)
    B = log_gamma(0.5 * dof)
    C = 0.5 * sample_dim * np.log(dof * np.pi)
    D = 0.5 * log_det_scale_matrix

    normalization_contribution = num_samples * (A - B - C - D)
    sample_contributions = -exponent * np.log1p(mahalanobis_squared_distances / dof)
    total_log_likelihood = normalization_contribution + sample_contributions.sum()

    return total_log_likelihood


@njit
def _mv_t_ll_at_mle_params(
    X: np.ndarray,
    start: int,
    end: int,
    dof: float,
) -> float:
    """Calculate multivariate T log likelihood at the MLE parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the maximum likelihood parameter
        estimates for the mean and scale matrix, given
        the provided degrees of freedom.
    """
    X_segment = X[start:end]

    # Estimate the mean of each dimension through the sample medians:
    sample_medians = col_median(X_segment)
    X_centered = X_segment - sample_medians

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(X_centered, dof)

    total_log_likelihood = _multivariate_t_log_likelihood(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, dof=dof
    )

    return total_log_likelihood


@njit
def multivariate_t_cost_mle_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, dof: float
) -> np.ndarray:
    """Calculate the multivariate T twice negative log likelihood cost.

    At the maximum likelihood estimated mean and scale matrix values.

    Parameters
    ----------
    starts : np.ndarray
        The start indices of the segments.
    ends : np.ndarray
        The end indices of the segments.
    X : np.ndarray
        The data matrix.
    dof : float
        The degrees of freedom for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        The twice negative log likelihood costs for each segment.
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))

    for i in prange(num_starts):
        segment_log_likelihood = _mv_t_ll_at_mle_params(X, starts[i], ends[i], dof=dof)
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


@njit
def _mv_t_ll_at_fixed_params(
    X: np.ndarray,
    start: int,
    end: int,
    mean: np.ndarray,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
    dof: float,
) -> float:
    """Calculate multivariate T log likelihood at the MLE parameters for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).
    mean : np.ndarray
        The mean of the multivariate t-distribution.
    inverse_scale_matrix : np.ndarray
        The inverse of the scale matrix of the multivariate t-distribution.
    log_det_scale_matrix : float
        The log determinant of the scale matrix of the multivariate t-distribution.
    dof : float
        The degrees of freedom of the multivariate t-distribution.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the observations in the
        interval ``[start, end)`` in the data matrix `X`,
        evaluated at the maximum likelihood parameter
        estimates for the mean and scale matrix, given
        the provided degrees of freedom.
    """
    X_centered = X[start:end] - mean

    # Compute the log likelihood of the segment:
    total_log_likelihood = _multivariate_t_log_likelihood(
        inverse_scale_matrix=inverse_scale_matrix,
        log_det_scale_matrix=log_det_scale_matrix,
        centered_samples=X_centered,
        dof=dof,
    )

    return total_log_likelihood


@njit
def multivariate_t_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    inverse_scale_matrix: np.ndarray,
    log_det_scale_matrix: float,
    dof: float,
) -> np.ndarray:
    """Calculate the multivariate T twice negative log likelihood cost.

    At fixed parameter values.

    Parameters
    ----------
    starts : np.ndarray
        The start indices of the segments.
    ends : np.ndarray
        The end indices of the segments.
    X : np.ndarray
        The data matrix.
    mv_t_mean : np.ndarray
        The fixed mean for the cost calculation.
    mv_t_scale_matrix : np.ndarray
        The fixed scale matrix for the cost calculation.
    mv_t_dof : float
        The fixed degrees of freedom for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        The twice negative log likelihood costs for each segment.
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))

    for i in prange(num_starts):
        segment_log_likelihood = _mv_t_ll_at_fixed_params(
            X,
            starts[i],
            ends[i],
            mean=mean,
            inverse_scale_matrix=inverse_scale_matrix,
            log_det_scale_matrix=log_det_scale_matrix,
            dof=dof,
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


@njit
def isotropic_t_dof_estimate(
    centered_samples: np.ndarray, zero_norm_tol=1.0e-6, infinite_dof_threshold=1.0e2
) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    sample_dim = centered_samples.shape[1]

    squared_norms = np.sum(centered_samples * centered_samples, axis=1)

    log_norm_sq_var = np.log(squared_norms[squared_norms > zero_norm_tol**2]).var()

    b = log_norm_sq_var - trigamma(sample_dim / 2.0)
    inf_dof_b_threshold = (2 * infinite_dof_threshold + 4) / (infinite_dof_threshold**2)
    if b < inf_dof_b_threshold:
        # The dof estimate formula would exceed the infinite dof threshold,
        # (or break down due to a negative value), so we return infinity.
        return np.inf

    dof_estimate = (1 + np.sqrt(1 + 4 * b)) / b

    return dof_estimate


@njit
def kurtosis_t_dof_estimate(
    centered_samples: np.ndarray, infinite_dof_threshold: float = 1.0e2
) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution."""
    sample_ellipitical_kurtosis = kurtosis(centered_samples).mean() / 3.0

    inf_dof_kurtosis_threshold = 2.0 / (infinite_dof_threshold - 4.0)
    if sample_ellipitical_kurtosis < inf_dof_kurtosis_threshold:
        # The elliptical kurtosis estimate is below the threshold
        # which would lead to a degrees of freedom estimate above the
        # infinite degrees of freedom threshold. We return infinity.
        return np.inf

    dof_estimate = (2.0 / sample_ellipitical_kurtosis) + 4.0
    return dof_estimate


@njit
def iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float = 5.0e1,
    max_iter=10,
    rel_tol=5.0e-2,
    abs_tol=1.0e-1,
    inner_max_iter=50,
    inner_atol=1.0e-5,
) -> float:
    """Algorithm 1: Automatic data-adaptive computation of the d.o.f. parameter nu.

    From:
    Shrinking the eigenvalues of M-estimators of covariance matrix.
    """
    n = centered_samples.shape[0]
    if initial_dof > infinite_dof_threshold:
        return np.inf

    inf_dof_nu_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)

    sample_covariance = (centered_samples.T @ centered_samples) / n

    mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        initial_dof,
        max_iter=inner_max_iter,
        abs_tol=inner_atol,
    )

    dof = initial_dof
    for _ in range(max_iter):
        nu_i = np.trace(sample_covariance) / np.trace(mle_scale_matrix)
        if nu_i < inf_dof_nu_threshold:
            # The estimated degrees of freedom are high enough to approximate the
            # multivariate T distribution with a Gaussian.
            dof = np.inf
            break

        old_dof = dof
        dof = 2 * nu_i / max((nu_i - 1), 1.0e-3)

        mle_scale_matrix, inner_mle_iterations = solve_for_mle_scale_matrix(
            initial_scale_matrix=mle_scale_matrix,
            centered_samples=centered_samples,
            dof=dof,
            max_iter=inner_max_iter,
            abs_tol=inner_atol,
        )

        absolute_dof_diff = np.abs(dof - old_dof)
        rel_tol_satisfied = absolute_dof_diff / old_dof < rel_tol
        abs_tol_satisfied = absolute_dof_diff < abs_tol
        if rel_tol_satisfied or abs_tol_satisfied:
            break

    return dof


@njit
def loo_iterative_mv_t_dof_estimate(
    centered_samples: np.ndarray,
    initial_dof: float,
    infinite_dof_threshold: float = 1.0e2,
    max_iter=5,
    rel_tol=5.0e-2,
    abs_tol=1.0e-1,
    inner_max_iter=50,
    inner_abs_tol=1.0e-5,
) -> float:
    """Estimate the degrees of freedom of a multivariate T distribution.

    Using an improved estimator, based on the algorithm in:
    'Improved estimation of the degree of freedom parameter of mv t-distribution''.
    However, the algorithm computes one MLE scale matrix estimate per samples,
    holding out one sample at a time, which increases computation cost.
    """
    if initial_dof > infinite_dof_threshold:
        return np.inf

    num_samples, sample_dimension = centered_samples.shape
    inf_dof_theta_threshold = infinite_dof_threshold / (infinite_dof_threshold - 2.0)

    sample_covariance = (centered_samples.T @ centered_samples) / num_samples
    grand_mle_scale_matrix = maximum_likelihood_mv_t_scale_matrix(
        centered_samples,
        dof=initial_dof,
        max_iter=inner_max_iter,
        abs_tol=inner_abs_tol,
    )
    contraction_estimate = np.trace(grand_mle_scale_matrix) / np.trace(
        sample_covariance
    )

    loo_sample = np.zeros((sample_dimension, 1))
    loo_sample_outer_product = np.zeros((sample_dimension, sample_dimension))

    current_dof = initial_dof

    # For parallelization, need to copy the centered_samples per thread:
    for _ in range(max_iter):
        total_loo_mahalanobis_squared_distance = 0.0
        for sample in range(num_samples):
            # Extract the leave-one-out sample as a column vector:
            loo_sample[:] = centered_samples[sample, :].reshape(-1, 1)
            loo_sample_outer_product[:, :] = loo_sample @ loo_sample.T

            # Initial estimate of the leave-one-out covariance matrix,
            # subtracting the contracted contribution of the leave-one-out sample:
            loo_scale_estimate = grand_mle_scale_matrix - contraction_estimate * (
                loo_sample_outer_product / num_samples
            )

            # Zero out the leave-one-out sample:
            centered_samples[sample, :] = 0.0

            loo_mle_scale_matrix, inner_iters = solve_for_mle_scale_matrix(
                initial_scale_matrix=loo_scale_estimate,
                centered_samples=centered_samples,
                dof=current_dof,
                num_zeroed_samples=1,
                abs_tol=inner_abs_tol,
            )

            # Restore the leave-one-out sample:
            centered_samples[sample, :] = loo_sample[:].reshape(-1)

            loo_mahalanobis_squared_distance = (
                loo_sample.T @ np.linalg.solve(loo_mle_scale_matrix, loo_sample)
            )[0, 0]
            total_loo_mahalanobis_squared_distance += loo_mahalanobis_squared_distance

        theta_k = (1 - sample_dimension / num_samples) * (
            (total_loo_mahalanobis_squared_distance / num_samples) / sample_dimension
        )
        if theta_k < inf_dof_theta_threshold:
            # The estimated degrees of freedom are high enough to approximate the
            # multivariate T distribution with a Gaussian distribution.
            current_dof = np.inf
            break

        new_t_dof = 2 * theta_k / (theta_k - 1)
        abs_difference = np.abs(new_t_dof - current_dof)
        abs_tol_satisfied = abs_difference < abs_tol
        rel_tol_satisfied = (abs_difference / current_dof) < rel_tol

        current_dof = new_t_dof
        if abs_tol_satisfied or rel_tol_satisfied:
            break

    return current_dof


@njit
def estimate_mv_t_dof(
    X: np.ndarray, infinite_dof_threshold: float, refine_dof_threshold: int
):
    centered_samples = X - col_median(X)

    isotropic_dof = isotropic_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )
    kurtosis_dof = kurtosis_t_dof_estimate(
        centered_samples, infinite_dof_threshold=infinite_dof_threshold
    )

    if np.isfinite(isotropic_dof) and np.isfinite(kurtosis_dof):
        # Initialize the iterative dof estimation method with the
        # geometric mean of the isotropic and kurtosis estimates:
        initial_dof_estimate = np.sqrt(isotropic_dof * kurtosis_dof)
    elif np.isfinite(isotropic_dof):
        initial_dof_estimate = isotropic_dof
    elif np.isfinite(kurtosis_dof):
        initial_dof_estimate = kurtosis_dof
    else:
        # Both initial estimates are infinite, start the
        # iterative method with a reasonably high initial dof:
        initial_dof_estimate = infinite_dof_threshold / 2.0

    dof_estimate = iterative_mv_t_dof_estimate(
        centered_samples=centered_samples,
        initial_dof=initial_dof_estimate,
        infinite_dof_threshold=infinite_dof_threshold,
    )

    num_samples = X.shape[0]
    if num_samples <= refine_dof_threshold:
        dof_estimate = loo_iterative_mv_t_dof_estimate(
            centered_samples=centered_samples,
            initial_dof=dof_estimate,
            infinite_dof_threshold=infinite_dof_threshold,
        )

    return dof_estimate


class MultivariateTCost(BaseCost):
    r"""Multivariate T likelihood cost.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean and scale matrix for the cost calculation.
        If ``None``, the maximum likelihood estimates are used.
    dof : float, optional (default=None)
        Fixed degrees of freedom for the cost calculation.
        If None, the degrees of freedom are estimated from the data.
    infinite_dof_threshold : float, optional (default=1.0e2)
        The threshold at which the degrees of freedom are considered infinite.
        If the degrees of freedom are above this threshold,
        the multivariate t-distribution is approximated with
        a multivariate Gaussian distribution.
    refine_dof_threshold : int, optional (default=500)
        The number of samples below which the degrees of freedom
        estimate is refined using a leave-one-out iterative method.
    """

    evaluation_type = "multivariate"

    def __init__(
        self,
        param: Union[tuple[MeanType, CovType], None] = None,
        dof=None,
        infinite_dof_threshold=1.0e2,
        refine_dof_threshold=500,
    ):
        super().__init__(param)
        self.dof = dof
        self.infinite_dof_threshold = infinite_dof_threshold
        self.refine_dof_threshold = refine_dof_threshold

    def _check_fixed_param(
        self, param: tuple[MeanType, CovType], X: np.ndarray
    ) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : 2-tuple of float or np.ndarray
            Fixed mean and covariance matrix for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean, cov = param
        mean = check_mean(mean, X)
        cov = check_cov(cov, X)
        return mean, cov

    @property
    def min_size(self) -> Union[int, None]:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        """
        if self.is_fitted:
            return self._X.shape[1] + 1
        else:
            return None

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return 1 + p + p * (p + 1) // 2

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method checks fixed distribution parameters, if provided, and
        precomputes quantities that are used in the cost evaluation.

        Additionally, the degrees of freedom for the multivariate T data generating
        distribution it estimated, if the degrees of freedom parameters was not set
        during the construction of the MultivariateTCost object.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._param = self._check_param(self.param, X)

        if self.param is not None:
            self._mean, scale_matrix = self._param
            self._inv_scale_matrix = np.linalg.inv(scale_matrix)
            _, self._log_det_scale_matrix = np.linalg.slogdet(scale_matrix)

        if self.dof is None:
            self.dof_ = estimate_mv_t_dof(
                X,
                infinite_dof_threshold=self.infinite_dof_threshold,
                refine_dof_threshold=self.refine_dof_threshold,
            )
        else:
            self.dof_ = self.dof

        if not np.isposinf(self.dof_) and (
            self.dof_ <= 0.0 or not np.isfinite(self.dof_)
        ):
            raise ValueError(
                "Degrees of freedom 'dof' must be a positive,"
                " finite number, or 'np.inf'."
            )

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the MLE parameters.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is 1 since the MultivariateTCost is inherently multivariate.
        """
        if np.isposinf(self.dof_):
            return gaussian_cost_mle_params(starts, ends, X=self._X)
        else:
            return multivariate_t_cost_mle_params(
                starts, ends, X=self._X, dof=self.dof_
            )

    def _evaluate_fixed_param(self, starts, ends):
        """Evaluate the cost for the fixed parameters.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of columns
            is 1 since the MultivariateGaussianCost is inherently multivariate.
        """
        if np.isposinf(self.dof):
            return gaussian_cost_fixed_params(
                starts,
                ends,
                self._X,
                self._mean,
                self._log_det_scale_matrix,
                self._inv_scale_matrix,
            )
        else:
            return multivariate_t_cost_fixed_params(
                starts,
                ends,
                self._X,
                mean=self._mean,
                inverse_scale_matrix=self._inv_scale_matrix,
                log_det_scale_matrix=self._log_det_scale_matrix,
                dof=self.dof,
            )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for interval evaluators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params = [
            {"param": None},
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.eye(1))},
        ]
        return params
