"""Multivariate T distribution likelihood cost."""

__author__ = ["johannvk"]
__all__ = ["MultivariateTCost"]

from typing import Union

import numpy as np
import scipy.linalg as sla
import scipy.stats as st
from scipy.special import digamma, polygamma

from skchange.costs.base import BaseCost
from skchange.costs.utils import CovType, MeanType, check_cov, check_mean
from skchange.utils.numba import jit, njit, prange
from skchange.utils.numba.stats import log_det_covariance


def estimate_mle_cov_scale(centered_samples: np.ndarray, dof: float):
    """Estimate the scale parameter of the MLE covariance matrix."""
    p = centered_samples.shape[1]
    squared_norms = np.sum(centered_samples * centered_samples, axis=1)
    z_bar = np.log(squared_norms[squared_norms > 1.0e-12]).mean()
    log_alpha = z_bar - np.log(dof) + digamma(0.5 * dof) - digamma(p / 2.0)
    return np.exp(log_alpha)


def initial_scale_matrix_estimate(
    centered_samples: np.ndarray, t_dof: float, num_zeroed_samples: int = 0
):
    """Estimate the scale matrix given centered samples and degrees of freedom."""
    n, p = centered_samples.shape
    num_effective_samples = n - num_zeroed_samples

    sample_covariance_matrix = (
        centered_samples.T @ centered_samples
    ) / num_effective_samples

    alpha_estimate = estimate_mle_cov_scale(centered_samples, t_dof)
    contraction_estimate = alpha_estimate * (p / np.trace(sample_covariance_matrix))
    scale_matrix_estimate = contraction_estimate * sample_covariance_matrix

    return scale_matrix_estimate


@jit(nopython=False)
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

    inv_cov_2d = sla.solve(
        scale_matrix, np.eye(p), assume_a="pos", overwrite_a=False, overwrite_b=True
    )
    z_scores = np.einsum("ij,jk,ik->i", centered_samples, inv_cov_2d, centered_samples)

    sample_weight = (p + t_dof) / (t_dof + z_scores)
    weighted_samples = centered_samples * sample_weight[:, np.newaxis]

    reconstructed_scale_matrix = (
        weighted_samples.T @ centered_samples
    ) / effective_num_samples

    return reconstructed_scale_matrix


def solve_mle_scale_matrix(
    initial_scale_matrix: np.ndarray,
    centered_samples: np.ndarray,
    t_dof: float,
    num_zeroed_samples: int = 0,
    max_iter: int = 50,
    reverse_tol: float = 1.0e-3,
) -> np.ndarray:
    """Perform fixed point iterations for the MLE scale matrix."""
    scale_matrix = initial_scale_matrix.copy()
    temp_cov_matrix = initial_scale_matrix.copy()

    # Compute the MLE covariance matrix using fixed point iteration:
    for iteration in range(max_iter):
        temp_cov_matrix = scale_matrix_fixed_point_iteration(
            scale_matrix=scale_matrix,
            t_dof=t_dof,
            centered_samples=centered_samples,
            num_zeroed_samples=num_zeroed_samples,
        )
        residual = sla.norm(temp_cov_matrix - scale_matrix, ord="fro")

        scale_matrix = temp_cov_matrix.copy()
        if residual < reverse_tol:
            break

    return scale_matrix, iteration


def maximum_likelihood_scale_matrix(
    centered_samples: np.ndarray,
    t_dof: float,
    reverse_tol: float = 1.0e-3,
    max_iter: int = 50,
    num_zeroed_samples: int = 0,
) -> np.ndarray:
    """Compute the MLE covariance matrix for a multivariate t-distribution.

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
    # Initialize the covariance matrix:
    mle_scale_matrix = initial_scale_matrix_estimate(
        centered_samples, t_dof, num_zeroed_samples=num_zeroed_samples
    )

    mle_scale_matrix, inner_iterations = solve_mle_scale_matrix(
        initial_scale_matrix=mle_scale_matrix,
        centered_samples=centered_samples,
        t_dof=t_dof,
        num_zeroed_samples=num_zeroed_samples,
        max_iter=max_iter,
        reverse_tol=reverse_tol,
    )

    return mle_scale_matrix


def _mv_t_ll_at_mle_params(
    X: np.ndarray,
    start: int,
    end: int,
    t_dof: float,
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
    t_dof : float
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

    # Compute the MLE scale matrix:
    mle_scale_matrix = maximum_likelihood_scale_matrix(X_centered, t_dof)

    # Compute the log likelihood of the segment:
    mv_t_dist = st.multivariate_t(loc=sample_medians, shape=mle_scale_matrix, df=t_dof)
    log_likelihood = mv_t_dist.logpdf(X_segment).sum()

    return log_likelihood


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
            X, starts[i], ends[i], t_dof=mv_t_dof
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


def _mv_t_ll_at_fixed_params(
    X: np.ndarray,
    start: int,
    end: int,
    mv_t_mean: np.ndarray,
    mv_t_scale_matrix: np.ndarray,
    mv_t_dof: float,
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
    mv_t_mean : np.ndarray
        The mean of the multivariate t-distribution.
    mv_t_scale_matrix : np.ndarray
        The scale matrix of the multivariate t-distribution.
    mv_t_dof : float
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

    # Compute the log likelihood of the segment:
    mv_t_dist = st.multivariate_t(loc=mv_t_mean, shape=mv_t_scale_matrix, df=mv_t_dof)
    log_likelihood = mv_t_dist.logpdf(X_segment).sum()

    return log_likelihood


def multivariate_t_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mv_t_mean: np.ndarray,
    mv_t_scale_matrix: np.ndarray,
    mv_t_dof: float,
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
            X, starts[i], ends[i], mv_t_mean, mv_t_scale_matrix, mv_t_dof=mv_t_dof
        )
        costs[i, 0] = -2.0 * segment_log_likelihood

    return costs


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

    b = log_norm_sq_var - polygamma(1, sample_dim / 2.0)
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
