"""Numba-optimized functions for various common data manipulation tasks.""" ""

import numpy as np
from numba import njit, prange


@njit
def col_repeat(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat each column of a 2D array n times.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    """
    expanded_x = np.zeros((x.shape[0], n))
    for j in prange(n):
        expanded_x[:, j] = x
    return expanded_x


@njit
def row_repeat(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat each row of a 2D array n times.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    """
    expanded_x = np.zeros((x.shape[0], n))
    for i in prange(n):
        expanded_x[i, :] = x
    return expanded_x
