"""Test statistic for differences in the variance."""

from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.stats import col_cumsum


@njit
def init_var_score(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute sums for mean_score.

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
    sums2 = np.zeros((n + 1, p))
    sums2[1:] = col_cumsum(X**2)
    return sums2


@njit
def var_score(
    precomputed_params: np.ndarray, start: int, end: int, split: int
) -> float:
    """Calculate the score for a change in variance.

    Compares the variance of the data before and after the split within the interval
    from start:end.

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from init_var_score.
    start : int
        Start index of the interval. Must be < end, split.
    end : int
        End index of the interval. Must be > start, split
    split : int
        Split index of the interval. Must be > start and < end.

    Returns
    -------
    stat : float
        Score for a difference in the variance.

    Notes
    -----
    To optimize performance, no checks are performed on (start, split, end).
    """
    sums2 = precomputed_params
    before_sum2 = sums2[split + 1] - sums2[start]
    before_weight = np.sqrt((end - split) / (end - start + 1) * (split - start + 1))
    after_sum2 = sums2[end + 1] - sums2[split + 1]
    after_weight = np.sqrt((split - start + 1) / (end - start + 1) * (end - split))
    return np.sum(np.abs(after_weight * after_sum2 - before_weight * before_sum2))
