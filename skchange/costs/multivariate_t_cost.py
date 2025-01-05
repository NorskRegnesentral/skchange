"""Multivariate T distribution likelihood cost."""

__author__ = ["johannvk"]
__all__ = ["MultivariateTCost"]

from typing import Union

import numpy as np
import scipy.stats as st

from skchange.costs.base import BaseCost
from skchange.utils.numba import njit, prange


@njit
def numba_log_gamma(x: float) -> float:
    """Compute the log of the gamma function.

    Uses the Stirling's approximation for the gamma function.
    Source: https://en.wikipedia.org/wiki/Gamma_function#Log-gamma_function
    """
    x_cubed = x * x * x
    log_gamma = (
        (x - 0.5) * np.log(x)
        - x
        + 0.5 * np.log(2.0 * np.pi)
        + 1.0 / (12.0 * x)
        - 1.0 / (360.0 * x_cubed)
        + 1.0 / (1260.0 * x_cubed * x * x)
    )

    return log_gamma


@njit
def numba_digamma(x: float) -> float:
    """Approximate the digamma function.

    Use the asymptotic expansion for the digamma function on the real domain,
    by first moving the argument above 5.0 before
    applying the first three terms of its asymptotic expansion.

    Source: https://en.wikipedia.org/wiki/Digamma_function#Asymptotic_expansion
    """
    result = 0.0
    while x <= 5.0:
        result -= 1.0 / x
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += np.log(x) - 0.5 * inv_x - inv_x2 * (1.0 / 12.0 - inv_x2 / 120.0)
    return result


@njit
def numba_trigamma(x: float) -> float:
    """
    Approximate the trigamma function on the real positive domain.

    Uses the asymptotic expansion for the trigamma function on the real domain,
    by first moving the argument above 5.0 before
    applying the first four terms of its asymptotic expansion.

    Source: https://en.wikipedia.org/wiki/Trigamma_function
    """
    result = 0.0
    while x <= 5.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += (
        (1.0 / x)
        + 0.5 * inv_x2
        + (1.0 / 6.0) * inv_x2 * inv_x
        + (1.0 / 30.0) * inv_x2 * inv_x2 * inv_x
    )
    return result


@njit
def estimate_scale_matrix_trace(centered_samples: np.ndarray, dof: float):
    """Estimate the scale parameter of the MLE covariance matrix.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    p = centered_samples.shape[1]
    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    z_bar = np.log(squared_norms[squared_norms > 1.0e-12]).mean()
    log_alpha = z_bar - np.log(dof) + numba_digamma(0.5 * dof) - numba_digamma(p / 2.0)
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
    t_dof: float,
    centered_samples: np.ndarray,
    num_zeroed_samples: int = 0,
):
    """Compute the MLE covariance residual for a mv_t distribution."""
    n, p = centered_samples.shape

    # Subtract the number of 'zeroed' samples:
    effective_num_samples = n - num_zeroed_samples

    inverse_scale_matrix = np.linalg.solve(scale_matrix, np.eye(p))
    mahalanobis_squared_distances = np.sum(
        (centered_samples @ inverse_scale_matrix) * centered_samples, axis=1
    )

    sample_weights = (p + t_dof) / (t_dof + mahalanobis_squared_distances)
    weighted_samples = centered_samples * sample_weights[:, np.newaxis]

    reconstructed_scale_matrix = (
        weighted_samples.T @ centered_samples
    ) / effective_num_samples

    return reconstructed_scale_matrix


@njit
def solve_mle_scale_matrix(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    max_iter: int = 50,
    reverse_tol: float = 1.0e-3,
) -> np.ndarray:
    """Perform fixed point iterations to compute the MLE scale matrix."""
    scale_matrix = initial_scale_matrix.copy()
    for iteration in range(max_iter):
        temp_cov_matrix = scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            t_dof=t_dof,
            centered_samples=centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )

        # Note: 'ord = None' computes the Frobenius norm.
        residual = np.linalg.norm(temp_cov_matrix - scale_matrix, ord=None)

        scale_matrix[:, :] = temp_cov_matrix[:, :]
        if residual < reverse_tol:
            break

    return scale_matrix, iteration


@njit
def maximum_likelihood_scale_matrix(
    centered_samples: np.ndarray,
    t_dof: float,
    reverse_tol: float = 1.0e-3,
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
        t_dof,
        num_zeroed_samples=num_zeroed_samples,
        apply_trace_correction=initial_trace_correction,
    )

    mle_scale_matrix, inner_iterations = solve_mle_scale_matrix(
        initial_scale_matrix=mle_scale_matrix,
        centered_samples=centered_samples,
        t_dof=t_dof,
        num_zeroed_samples=num_zeroed_samples,
        max_iter=max_iter,
        reverse_tol=reverse_tol,
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
    A = numba_log_gamma(exponent)
    B = numba_log_gamma(0.5 * dof)
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
    sample_medians = np.median(X_segment, axis=0)
    X_centered = X_segment - sample_medians

    mle_scale_matrix = maximum_likelihood_scale_matrix(X_centered, dof)

    total_log_likelihood = _multivariate_t_log_likelihood(
        scale_matrix=mle_scale_matrix, centered_samples=X_centered, dof=dof
    )

    return total_log_likelihood


@njit
def multivariate_t_cost_mle_params(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, mv_t_dof: float
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
    mv_t_dof : float
        The degrees of freedom for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        The twice negative log likelihood costs for each segment.
    """
    num_starts = len(starts)
    costs = np.zeros((num_starts, 1))

    for i in prange(num_starts):
        segment_log_likelihood = _mv_t_ll_at_mle_params(
            X, starts[i], ends[i], dof=mv_t_dof
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


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
def estimate_mv_t_dof(centered_samples: np.ndarray, zero_norm_tol=1.0e-6) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution.

    From: A Novel Parameter Estimation Algorithm for the Multivariate
          t-Distribution and Its Application to Computer Vision.
    """
    sample_dim = centered_samples.shape[1]

    squared_norms = np.sum(centered_samples * centered_samples, axis=1)

    log_norm_sq: np.ndarray
    log_norm_sq = np.log(squared_norms[squared_norms > zero_norm_tol**2])
    log_norm_sq_var = log_norm_sq.var(ddof=0)

    b = log_norm_sq_var - numba_trigamma(sample_dim / 2.0)
    t_dof_estimate = (1 + np.sqrt(1 + 4 * b)) / b

    return t_dof_estimate


def ellipitical_kurtosis(centered_samples: np.ndarray) -> float:
    """Compute the kurtosis of a set of samples.

    From:
    Shrinking the eigenvalues of M-estimators of covariance matrix,
    subsection IV-A.
    by: Esa Ollila, Daniel P. Palomar, and Frédéric Pascal
    """
    kurtosis_per_sample_dim: np.ndarray
    kurtosis_per_sample_dim = st.kurtosis(
        centered_samples, fisher=True, bias=False, axis=0
    )
    averaged_kurtosis = kurtosis_per_sample_dim.mean()
    return averaged_kurtosis / 3.0


def kurtosis_t_dof_estimate(centered_samples: np.ndarray) -> float:
    """Estimate the degrees of freedom of a multivariate t-distribution."""
    sample_ellipitical_kurtosis = ellipitical_kurtosis(centered_samples)
    t_dof_estimate = 2.0 / max(1.0e-3, sample_ellipitical_kurtosis) + 4.0
    return t_dof_estimate


class MultivariateTCost(BaseCost):
    pass
