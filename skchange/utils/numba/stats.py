"""Numba-optimized functions for calculating various statistics."""

import numpy as np
from numba import njit


@njit
def col_cumsum(x: np.ndarray):
    """Calculate the cumulative sum of each column in a 2D array.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    """
    cumsum = np.zeros_like(x)
    for j in range(x.shape[1]):
        cumsum[:, j] = np.cumsum(x[:, j])
    return cumsum
