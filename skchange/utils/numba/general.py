"""Numba-optimized functions for various common data manipulation tasks."""

import numpy as np

from skchange.utils.numba import njit, prange


@njit
def col_repeat(x: np.ndarray, n: int) -> np.ndarray:
    """Repeat each column of a 2D array n times.

    Parameters
    ----------
    x : np.ndarray
        1D array.

    Returns
    -------
    2D array : (x.size, n)-matrix with x in each column
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
        1D array.

    Returns
    -------
    2D array : (n, x.size) matrix with x in each row
    """
    expanded_x = np.zeros((x.shape[0], n))
    for i in prange(n):
        expanded_x[i, :] = x
    return expanded_x


@njit
def where(indicator: np.ndarray) -> list:
    """
    Identify consecutive intervals of True values in the input array.

    Parameters
    ----------
    indicator : np.ndarray
        1D boolean array.

    Returns
    -------
    list of tuples:
        Each tuple represents the start and end indices of consecutive True intervals.
        If there are no True values, an empty list is returned.
    """
    intervals = []
    start, end = None, None
    for i, val in enumerate(indicator):
        if val and start is None:
            start = i
        elif not val and start is not None:
            end = i - 1
            intervals.append((start, end))
            start, end = None, None
    if start is not None:
        intervals.append((start, len(indicator) - 1))
    return intervals


@njit
def truncate_below(x: np.ndarray, lower_bound: float) -> np.ndarray:
    """Truncate values below a lower bound.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    lower_bound : float
        Lower bound.

    Returns
    -------
    x : np.ndarray
        2D array with values below lower_bound replaced by lower_bound.
    """
    if x.ndim == 1:
        x[x < lower_bound] = lower_bound
    else:
        p = x.shape[1]
        for j in range(p):
            # Numba doesn't support multidimensional index.
            x[x[:, j] < lower_bound, j] = lower_bound  # To avoid zero division
    return x
