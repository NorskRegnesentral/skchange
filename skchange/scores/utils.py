"""Utility functions for score calculations."""

import numpy as np

from skchange.utils.numba import njit
from skchange.utils.numba.general import col_repeat
from skchange.utils.numba.stats import col_cumsum


@njit
def init_sums(X: np.ndarray) -> np.ndarray:
    """Precompute cumulative sums."""
    return col_cumsum(X, init_zero=True)


@njit
def init_sums2(X: np.ndarray) -> np.ndarray:
    """Precompute cumulative sums of squares."""
    return col_cumsum(X**2, init_zero=True)


@njit
def init_sample_sizes(X: np.ndarray) -> np.ndarray:
    """Precompute all subset sample sizes."""
    n = X.shape[0]
    p = X.shape[1]
    return col_repeat(np.arange(0, n + 1), p)
