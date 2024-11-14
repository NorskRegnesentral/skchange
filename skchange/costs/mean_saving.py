"""Mean saving for CAPA type anomaly detection."""

__author__ = ["Tveten"]

import numpy as np

from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.njit import njit
from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_mean_saving(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Precompute sums and weights for `mean_saving`.

    Parameters
    ----------
    X : `np.ndarray`
        2D array.

    Returns
    -------
    sums : `np.ndarray`
        Cumulative sums of `X`.
    weights : `np.ndarray`
        Weights for `sums2` in the cost calculation.
    """
    n = X.shape[0]
    p = X.shape[1]
    # 0.0 as first row to make calculations work also for start = 0
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    weights = col_repeat(np.arange(0, n + 1), p)
    return sums, weights


@njit(cache=True)
def mean_saving(
    precomputed_params: tuple[np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Gaussian mean likelihood saving for each segment.

    The mean_saving calculates the Gaussian likelihood ratio test statistic of the
    segment starting at `start` and ending at `end` having the maximum likelihod
    estimate of the mean versus zero mean.

    Parameters
    ----------
    precomputed_params : `tuple[np.ndarray, np.ndarray]`
        Precomputed parameters from `init_mean_saving`.
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    savings : `np.ndarray`
        2D array of savings for each segment (rows) and component (columns).
    """
    sums, weights = precomputed_params
    saving = (sums[ends + 1] - sums[starts]) ** 2 / weights[ends - starts + 1]
    return saving
