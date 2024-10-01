"""Test statistic for differences in the covariance structure through the operator norm for multivariate data.

Reference: Optimal covariance change point localization in high dimensions: Daren Wang, Yiyu and Alessandro Rinaldo.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def init_covariance_diff_opnorm_score(X: np.ndarray) -> np.ndarray:
    """Initialize the precision matrix change point detection."""
    # TODO: Should (could) compute "rolling" covariance matrices here?
    return X


@njit(cache=True)
def _covariance_diff_opnorm_score(X: np.ndarray, start: int, end: int, split: int) -> float:
    """Calculate the CUSUM score for a change in the covariance matrix operator norm.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the interval to test for a change in the covariance matrix (inclusive).
    end : int
        End index of the interval to test for a change in the covariance matrix (inclusive).
    split : int
        Split point of the interval to test for a change in the covariance matrix.
    """
    num_obs, dims = X.shape[0], X.shape[1]
    min_num_obs = dims * np.log(num_obs)

    # Assert that the test interval is wide enough, and the split point is 
    # sufficiently far from the start and end points:
    # assert end - start > 2*min_num_obs, "Interval is too short."
    # assert np.ceil(start + min_num_obs) < split < np.floor(end - min_num_obs),\
    #        "Split point is too close to the end of the test interval."

    pre_split_cov = np.cov(X[start:split], rowvar=False)
    post_split_cov = np.cov(X[split : (end + 1)], rowvar=False)

    full_interval_length = end + 1 - start
    pre_split_length = split - start
    post_split_length = end + 1 - split

    pre_split_scaling = np.sqrt(post_split_length / (full_interval_length * pre_split_length))
    post_split_scaling = np.sqrt(pre_split_length / (full_interval_length * post_split_length))

    cov_diff_operator = (pre_split_scaling * pre_split_cov) - (post_split_scaling * post_split_cov)

    # Compute the operator norm of the covariance difference operator:
    # Since the covariance matrix is symmetric, the operator norm is the largest eigenvalue.
    cov_diff_eigvals = np.linalg.eigvalsh(cov_diff_operator)
    cov_diff_opnorm = np.max(cov_diff_eigvals)

    return cov_diff_opnorm



@njit(cache=True)
def covariance_diff_opnorm_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the mean and covariance for.

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

    Source:
    - A Multivariate Change point Model for Change in Mean Vector and/or Covariance Structure: Detection of Isolated
    Systolic Hypertension (ISH). K.D. Zamba.

    Returns
    -------
    scores : np.ndarray
        Scores (-2 negative log likelihood) for each split segment.
    """
    # print("Not using pseudo-determinant for now.")
    # Data matrix, column variables:
    X = precomputed_params
    num_splits = len(splits)

    # Assume: 'start', 'end', and 'split' are 1D integer arrays,
    # of the same length.
    if not (len(starts) == len(ends) == num_splits):
        raise ValueError("Lengths of 'starts', 'ends', and 'splits' must be the same.")

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _covariance_diff_opnorm_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores
