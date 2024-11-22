"""Utility functions for cost calculations."""

import numbers
from typing import Union

import numpy as np
from numpy.typing import ArrayLike

MeanType = Union[ArrayLike, numbers.Number]
CovType = ArrayLike


def check_mean(mean: MeanType, X: np.ndarray) -> np.ndarray:
    """Check if the fixed mean parameter is valid.

    Parameters
    ----------
    mean : np.ndarray or float
        Fixed mean for the cost calculation.
    X : np.ndarray
        2d input data.

    Returns
    -------
    mean : np.ndarray
        Fixed mean for the cost calculation.
    """
    mean = np.array([mean]) if isinstance(mean, numbers.Number) else np.asarray(mean)
    if len(mean) != 1 and len(mean) != X.shape[1]:
        raise ValueError(f"mean must have length 1 or X.shape[1], got {len(mean)}.")
    return mean


def check_cov(cov: CovType, X: np.ndarray) -> np.ndarray:
    """Check if the fixed covariance matrix parameter is valid.

    Parameters
    ----------
    cov : np.ndarray
        Fixed covariance matrix for the cost calculation.
    X : np.ndarray
        2d input data.

    Returns
    -------
    cov : np.ndarray
        Fixed covariance matrix for the cost calculation.
    """
    cov = np.asarray(cov)
    p = X.shape[1]
    if cov.shape[0] != p or cov.shape[1] != p:
        raise ValueError(
            f"cov must have shape (X.shape[1], X.shape[1]), got {cov.shape}."
        )
    return cov
