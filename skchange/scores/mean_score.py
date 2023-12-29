"""Test statistic for differences in the mean."""

from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_mean_score(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    return sums


@njit(cache=True)
def mean_score(
    precomputed_params: np.ndarray, start: int, end: int, split: int
) -> float:
    """Calculate the CUSUM score for a change in the mean.

    Compares the mean of the data before and after the split within the interval from
    start:end (both inclusive).

    Parameters
    ----------
    precomputed_params : np.ndarray
        Precomputed parameters from init_mean_score.
    start : int
        Start index of the interval. Must be < end, split.
    end : int
        End index of the interval. Must be > start, split
    split : int
        Split index of the interval. Must be > start and < end.

    Returns
    -------
    stat : float
        Score for a difference in the mean.

    Notes
    -----
    To optimize performance, no checks are performed on (start, split, end).
    """
    sums = precomputed_params
    before_sum = sums[split + 1] - sums[start]
    before_weight = np.sqrt((end - split) / ((end - start + 1) * (split - start + 1)))
    after_sum = sums[end + 1] - sums[split + 1]
    after_weight = np.sqrt((split - start + 1) / ((end - start + 1) * (end - split)))
    return np.sum(np.abs(after_weight * after_sum - before_weight * before_sum))
