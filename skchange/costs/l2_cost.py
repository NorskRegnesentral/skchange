"""L2 cost function for change point detection."""

from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


@njit
def init_l2_cost(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute sums and weights for l2_cost.

    Parameters
    ----------
    X : np.ndarray
        2D array.

    Returns
    -------
    sums : np.ndarray
        Cumulative sums of X.
    sums2 : np.ndarray
        Cumulative sums of X**2.
    weights : np.ndarray
        Weights for sums2 in the cost calculation.
    """
    n = X.shape[0]
    p = X.shape[1]

    # 0.0 as first row to make calculations work also for start = 0
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    sums2 = np.zeros((n + 1, p))
    sums2[1:] = col_cumsum(X**2)
    weights = col_repeat(np.arange(0, n + 1), p)
    return sums, sums2, weights


@njit
def l2_cost(
    precomputed_params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Calculate the l2 cost for each segment.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from init_l2_cost.
    starts : np.ndarray
        Start indices of the segments.
    ends : np.ndarray
        End indices of the segments.

    Returns
    -------
    costs : np.ndarray
        Costs for each segment.
    """
    sums, sums2, weights = precomputed_params
    partial_sums = sums[ends + 1] - sums[starts]
    partial_sums2 = sums2[ends + 1] - sums2[starts]
    weights = weights[ends - starts + 1]
    costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
    return costs
