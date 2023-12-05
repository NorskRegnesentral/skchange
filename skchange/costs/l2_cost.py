from typing import Tuple

import numpy as np
from numba import njit

from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


@njit
def init_l2_cost(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = X.shape[0]
    p = X.shape[1]

    # 0.0 as first row to make calculations work also for start = 0
    sums = np.zeros((n + 1, p))
    sums[1:] = col_cumsum(X)
    sums2 = np.zeros((n + 1, p))
    sums2[1:] = col_cumsum(X**2)
    weights = col_repeat(np.arange(0, n + 1), p)
    return sums, sums2, weights


# @njit
# def l2_cost(
#     precomputed_params: Tuple[np.ndarray, np.ndarray, np.ndarray],
#     starts: list,
#     ends: list,
# ) -> np.ndarray:
#     if len(ends) == 1:
#         ends = np.repeat(ends, len(starts))

#     sums, sums2, weights = precomputed_params

#     costs = np.zeros(len(starts))
#     for i in prange(len(starts)):
#         start, end = starts[i], ends[i]
#         partial_sum = sums[end + 1] - sums[start]
#         partial_sum2 = sums2[end + 1] - sums2[start]
#         weight = weights[end - start + 1]
#         costs[i] = np.sum(partial_sum2 - partial_sum**2 / weight)
#     return costs


@njit
def l2_cost(
    precomputed_params: Tuple[np.ndarray, np.ndarray, np.ndarray],
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    sums, sums2, weights = precomputed_params
    partial_sums = sums[ends + 1] - sums[starts]
    partial_sums2 = sums2[ends + 1] - sums2[starts]
    weights = weights[ends - starts + 1]
    costs = np.sum(partial_sums2 - partial_sums**2 / weights, axis=1)
    return costs
