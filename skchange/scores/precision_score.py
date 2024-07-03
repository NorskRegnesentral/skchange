"""Precision matrix change point detection"""

import numpy as np
from numba import njit, jit
from sklearn.covariance import GraphicalLasso, log_likelihood

gl_solver = GraphicalLasso(
    alpha=1.0e-2, mode="cd", covariance="precomputed", max_iter=1000
)

def solve_gl(X: np.ndarray, alpha_scaler: float) -> np.ndarray:
    """Solve the GraphicalLasso problem."""
    alpha_orig = gl_solver.alpha
    gl_solver.alpha *= alpha_scaler
    gl_solver.fit(X)
    gl_solver.alpha = alpha_orig
    return gl_solver.precision_


@njit(cache=True)
def init_precision_score(X: np.ndarray) -> np.ndarray:
    """Initialize the precision matrix change point detection."""
    # TODO: Should compute "rolling" covariance matrices here?
    return X


@njit(cache=True)
def precision_log_likelihood(cov: np.ndarray, precision: np.ndarray) -> float:
    """Calculate the unscaled log-likelihood of the precision matrix."""
    sign, log_det = np.linalg.slogdet(precision)
    if sign <= 0:
        return np.inf

    return np.sum(precision * cov) - log_det

# @jit(cache=True)
def _precision_score(
    X: np.ndarray, start: int, end: int, split: int
) -> float:
    """Calculate the CUSUM score for a change in the precision matrix.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the precision matrix.
    end : int
        End index of the interval to test for a change in the precision matrix.
    split : int
        Split index of the interval to test for a change in the precision matrix.
    """
    # TODO: Should 'total_samples' be the 'global' total number of samples?
    total_samples = end - start + 1
    pre_split_fraction = (split - start) / total_samples
    post_split_fraction = (end - split + 1) / total_samples

    pre_split_cov = np.cov(X[start:split], rowvar=False)
    # Update alpha to account for the fraction of samples in the pre-split segment:
    pre_l1_reg_scaler = np.sqrt(1.0 / pre_split_fraction)
    try:
        pre_split_precision = solve_gl(pre_split_cov, pre_l1_reg_scaler)
    except:
        return -1.0

    post_split_cov = np.cov(X[split : (end + 1)], rowvar=False)
    # Update alpha to account for the fraction of samples in the post-split segment:
    post_l1_reg_scaler = np.sqrt(1.0 / post_split_fraction)
    try:
        # post_split_precision = gl_solver.fit(post_split_cov).precision_
        post_split_precision = solve_gl(post_split_cov, post_l1_reg_scaler)
    except:
        return -1.0

    complete_cov = np.cov(X[start : (end + 1)], rowvar=False)
    try:
        complete_precision = solve_gl(complete_cov, 1.0)
    except:
        return -1.0

    # Remove normalization constant from LogLikelihood:
    pre_split_loss = precision_log_likelihood(pre_split_cov, pre_split_precision) * pre_split_fraction
    post_split_loss = precision_log_likelihood(post_split_cov, post_split_precision) * post_split_fraction
    interval_loss = precision_log_likelihood(complete_cov, complete_precision) * 1.0

    return interval_loss - (pre_split_loss + post_split_loss)


# @jit(cache=True)
def precision_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the covariance.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from init_covariance_score.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the covariance.
    ends : np.ndarray
        End indices of the intervals to test for a change in the covariance.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the covariance.

    # TODO: How to pass parameters to the GraphicalLasso solver?

    Returns
    -------
    scores : np.ndarray
        Scores for each split segment.
    """
    # Data matrix, column variables:
    X = precomputed_params
    num_splits = len(splits)

    # Assume: 'start', 'end', and 'split' are 1D integer arrays,
    # of the same length.
    if not len(starts) == len(ends) == num_splits:
        raise ValueError("Lengths of 'starts', 'ends', and 'splits' must be the same.")

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _precision_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores
