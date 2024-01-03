"""Test statistic for differences in the mean and/or variance."""

from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_meanvar_score(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute sums and squared sums for 'meanvar_score'.

    Parameters
    ----------
    X : np.ndarray
        2D array.

    Returns
    -------
    sums : np.ndarray
        Cumulative sums of X.
    """
    n = X.shape[0]
    p = X.shape[1]
    # 0.0 as first row to make calculations work also for start = 0 in mean_score
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    sums2 = np.zeros((n + 1, p))
    sums2[1:] = col_cumsum(X**2)
    return sums, sums2


@njit(cache=True)
def var_from_sums(sums1: np.ndarray, sums2: np.ndarray, i: np.ndarray, j: np.ndarray):
    """Calculate variance from precomputed sums.

    Parameters
    ----------
    sums1 : np.ndarray
        Cumulative sums of X.
    sums2 : np.ndarray
        Cumulative sums of X**2.
    i : np.ndarray
        Start indices in the original data X.
    j : np.ndarray
        End indices in the original data X.

    Returns
    -------
    var : float
        Variance of X[i:j] (inclusive j).
    """
    n = (j - i + 1).reshape(-1, 1)
    sum1 = sums1[j + 1] - sums1[i]  # Indices is moved one step forward
    sum2 = sums2[j + 1] - sums2[i]  # Indices is moved one step forward
    var = sum2 / n - (sum1 / n) ** 2

    p = var.shape[1]
    lower_bound = 1e-16  # standard deviation lower bound of 1e-8
    for j in range(p):
        # Numba doesn't support multidimensional index.
        var[var[:, j] < lower_bound, j] = lower_bound  # To avoid zero division
    return var


@njit(cache=True)
def meanvar_score(
    precomputed_params: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    split: np.ndarray,
) -> np.ndarray:
    """Calculate the score for a change in the mean and/or variance.

    Computes the likelihood ratio test for a change in the mean and/or variance of
    i.i.d. Gaussian data.

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from init_mean_score.
    start : np.ndarray
        Start indices of the intervals to test for a change in the mean.
    end : np.ndarray
        End indices of the intervals to test for a change in the mean.
    split : np.ndarray
        Split indices of the intervals to test for a change in the mean.

    Returns
    -------
    score : np.ndarray
        Scores for a difference in the mean or variance at the given intervals and
        splits.

    Notes
    -----
    To optimize performance, no checks are performed on (start, split, end).
    """
    sums, sums2 = precomputed_params
    before_var = var_from_sums(sums, sums2, start, split)
    after_var = var_from_sums(sums, sums2, split + 1, end)
    full_var = var_from_sums(sums, sums2, start, end)
    before_n = (split - start + 1).reshape(-1, 1)
    after_n = (end - split).reshape(-1, 1)
    before_term = -before_n * np.log(before_var / full_var)
    after_term = -after_n * np.log(after_var / full_var)
    likelihood_ratio = before_term + after_term
    return np.sum(likelihood_ratio, axis=1)
