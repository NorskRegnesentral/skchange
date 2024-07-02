"""Covariance change point detection"""

from typing import Tuple

import numpy as np
from numba import njit

@njit(cache=True)
def init_covariance_score(X: np.ndarray) -> np.ndarray:
    return X

@njit(cache=True)
def _covariance_score(X: np.ndarray, start: int, end: int, split: int) -> float:
    pre_split_cov = np.cov(X[start:split], rowvar=False)
    post_split_cov = np.cov(X[split:(end+1)], rowvar=False)
    score = np.linalg.norm(pre_split_cov - post_split_cov)
    return score


@njit(cache=True)
def covariance_score(
    precomputed_params: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    splits: np.ndarray,
) -> np.ndarray:
    """Calculate the CUSUM score for a change in the covariance.

    Parameters
    ----------
    precomputed_params : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Precomputed parameters from init_covariance_score.
    starts : np.ndarray
        Start indices of the intervals to test for a change in the covariance.
    ends : np.ndarray
        End indices of the intervals to test for a change in the covariance.
    splits : np.ndarray
        Split indices of the intervals to test for a change in the covariance.

    Returns
    -------
    scores : np.ndarray
        Scores for each split segment.
    """
    # Data matrix, column variables:
    X = precomputed_params

    num_splits = len(splits)

    # Assume: 'start', 'end', and 'split' are 1D integer arrays,
    # of the same length.
    if not len(starts) == len(ends) == num_splits:
        raise ValueError(
            "Lengths of 'starts', 'ends', and 'splits' must be the same."
        )

    scores = np.zeros(num_splits, dtype=np.float64)
    for split_idx in range(num_splits):
        scores[split_idx] = _covariance_score(
            X, starts[split_idx], ends[split_idx], splits[split_idx]
        )

    return scores
