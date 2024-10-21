"""Utility functions for detector components."""

import numpy as np
from numba import njit

from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


@njit(cache=True)
def init_sums(X: np.ndarray) -> np.ndarray:
    """Precompute cumulative sums."""
    return col_cumsum(X, init_zero=True)


@njit(cache=True)
def init_sums2(X: np.ndarray) -> np.ndarray:
    """Precompute cumulative sums of squares."""
    return col_cumsum(X**2, init_zero=True)


@njit(cache=True)
def init_sample_sizes(X: np.ndarray) -> np.ndarray:
    """Precompute all subset sample sizes."""
    n = X.shape[0]
    p = X.shape[1]
    return col_repeat(np.arange(0, n + 1), p)


@njit(cache=True)
def identity_func(X: np.ndarray) -> np.ndarray:
    """Identity function."""
    return X
