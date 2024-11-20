"""Numba-optimized functions for calculating various statistics."""

import numpy as np

from skchange.utils.numba import njit


@njit
def col_cumsum(x: np.ndarray, init_zero: bool = False) -> np.ndarray:
    """Calculate the cumulative sum of each column in a 2D array.

    Parameters
    ----------
    x : np.ndarray
        2D array.
    init_zero : bool
        Whether to let the first row be a row of zeros before the summing is
        started or not.

    Returns
    -------
    np.ndarray : Cumulative sums. If init_zero, the output contains one more
        row compared to the input x.

    """
    n = x.shape[0]
    p = x.shape[1]
    if init_zero:
        sums = np.zeros((n + 1, p))
        start = 1
    else:
        sums = np.zeros((n, p))
        start = 0

    for j in range(p):
        sums[start:, j] = np.cumsum(x[:, j])

    return sums


@njit
def log_det_covariance(X: np.ndarray) -> float:
    """Compute log determinant of the covariance matrix of a data matrix.

    Parameters
    ----------
    X : np.ndarray
        2D array of shape (n, p) where n is the number of samples and p is the number of
        variables.

    Returns
    -------
    log_abs_det : float
        The log of the absolute value of the determinant of the covariance matrix.
        Returns np.nan if the covariance matrix is not positive definite.

    """
    p = X.shape[1]
    cov = np.cov(X, rowvar=False, ddof=0).reshape(p, p)
    det_sign, log_abs_det = np.linalg.slogdet(cov)

    if det_sign <= 0:
        return np.nan
    else:
        return log_abs_det
