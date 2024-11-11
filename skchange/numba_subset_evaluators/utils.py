"""Utility functions for detector components."""

import numpy as np
from numba import njit


@njit(cache=True)
def identity(X: np.ndarray) -> np.ndarray:
    """Identity function."""
    return X


@njit(cache=True)
def subset_interval(X: np.ndarray, subsetter: np.ndarray) -> np.ndarray:
    """Subset an interval of an array along the first axis."""
    if len(subsetter) != 2:
        raise ValueError("The subsetter for costs must have two elements.")

    start = subsetter[0]
    end = subsetter[1]

    if start > end:
        raise ValueError("The start index must be less than or equal to the end index.")

    return X[start:end]


@njit(cache=True)
def split_intervals(X: np.ndarray, subsetter: np.ndarray) -> list[np.ndarray]:
    """Subset an interval of an array and split it into multiple intervals."""
    if len(subsetter) < 3:
        raise ValueError(
            "The subsetter for `split_interval` must have at least three elements."
        )

    if not np.all(np.diff(subsetter) > 0):
        raise ValueError("The subsetter indices must be in strictly increasing order.")

    start = subsetter[0]
    end = subsetter[-1]
    subsets = [X[start:end]]
    for i in range(1, len(subsetter)):
        X_sub = X[subsetter[i - 1] : subsetter[i]]
        subsets.append(X_sub)

    return subsets
