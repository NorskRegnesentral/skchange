"""Numba-optimized functions for calculating various statistics."""

import numpy as np
from numba import njit


@njit(cache=True)
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


@njit(cache=True)
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
