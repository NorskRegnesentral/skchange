"""Gaussian mean likelihood cost function for change point detection."""

__author__ = ["johannvk"]

import numpy as np
from numba import njit

from skchange.scores.mean_cov_score import _mean_cov_log_det_term


@njit(cache=True)
def init_mean_cov_cost(X: np.ndarray) -> np.ndarray:
    """Pass on the data matrix for the cost function."""
    return X


@njit(cache=True)
def gaussian_ll_at_mle_for_segment(
    X: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Calculate the Gaussian log likelihood at the MLE for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment.
    end : int
        End index of the segment.
    split : int
        Split index of the segment.

    Returns
    -------
    mv_ll_at_mle : float
        Log likelihood of the inclusive interval
        [start, end] in the data matrix X,
        evaluated at the maximum likelihood parameter
        estimates for the mean and covariance matrix.
    """
    if not end > start:
        raise ValueError(
            f"The 'end={end}' argument must be larger than 'start={start}'."
            " Cannot compute a covariance matrix from a single observation."
        )

    n = end - start + 1
    p = X.shape[1]

    n_times_log_det_cov = _mean_cov_log_det_term(X, start, end)
    mv_ll_at_mle = -((n * p) / 2) * np.log(2 * np.pi) - n_times_log_det_cov / 2.0

    # Subtract log(exp(Quadratic-form)) = p * n / 2, at MLE.
    mv_ll_at_mle -= (1 / 2) * p * n

    return mv_ll_at_mle


@njit(cache=True)
def mean_cov_cost(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    min_segment_length: int = 5,
) -> np.ndarray:
    """Calculate the Gaussian log likelihood cost for each segment.

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from `init_mean_cov_cost`.
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.

    Returns
    -------
    costs : np.ndarray
        Costs of the [start, end] segments, i.e., negative log likelihoods.
    """
    X = precomputed_params
    num_starts = len(starts)
    costs = np.zeros(num_starts)

    for i in range(num_starts):
        # TODO: DEBUG!
        if ends[i] - starts[i] + 1 < min_segment_length:
            costs[i] = np.inf
        else:
            segment_mv_ll = gaussian_ll_at_mle_for_segment(X, starts[i], ends[i])
            costs[i] = -segment_mv_ll
    return costs
