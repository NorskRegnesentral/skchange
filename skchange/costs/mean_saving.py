"""Mean saving for CAPA type anomaly detection."""

from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


@njit
def init_mean_saving(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute sums and weights for mean_cost.

    Parameters
    ----------
    X : np.ndarray
        2D array.

    Returns
    -------
    sums : np.ndarray
        Cumulative sums of X.
    weights : np.ndarray
        Weights for sums2 in the cost calculation.
    """
    n = X.shape[0]
    p = X.shape[1]
    # 0.0 as first row to make calculations work also for start = 0
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    weights = col_repeat(np.arange(0, n + 1), p)
    return sums, weights


@njit
def mean_saving(
    precomputed_params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Calculate the Gaussian mean likelihood saving for each segment.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray]
        Precomputed parameters from init_mean_saving.
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.

    Returns
    -------
    savings : np.ndarray
        2D array of savings for each segment (rows) and component (columns).
    """
    sums, weights = precomputed_params
    saving = (sums[ends + 1] - sums[starts]) ** 2 / weights[ends - starts + 1]
    return saving
