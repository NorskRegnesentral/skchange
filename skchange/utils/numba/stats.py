import numpy as np
from numba import njit


@njit
def col_cumsum(x: np.ndarray):
    cumsum = np.zeros_like(x)
    for j in range(x.shape[1]):
        cumsum[:, j] = np.cumsum(x[:, j])
    return cumsum
