"""Segment-label-based evaluation metrics.

Functions in this module take **dense per-sample label arrays** as their native
input — the integer arrays returned by ``predict()``.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score as _adjusted_rand_score
from sklearn.metrics import rand_score as _rand_score


def rand_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the Rand index for two segmentations.

    Measures the similarity between two segmentations by comparing all pairs of
    samples. Wraps ``sklearn.metrics.rand_score``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels, as returned by ``predict()``.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels, as returned by ``predict()``.

    Returns
    -------
    float
        Rand index in [0, 1]. Higher is better.

    Examples
    --------
    >>> rand_index(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]))
    1.0
    """
    return float(_rand_score(y_true, y_pred))


def adjusted_rand_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the adjusted Rand index for two segmentations.

    Similar to Rand index but adjusted for chance. Wraps
    ``sklearn.metrics.adjusted_rand_score``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels, as returned by ``predict()``.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels, as returned by ``predict()``.

    Returns
    -------
    float
        Adjusted Rand index in [-1, 1]. Higher is better.
        1.0 = perfect agreement, ~0.0 = random labeling.

    Examples
    --------
    >>> adjusted_rand_index(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]))
    1.0
    """
    return float(_adjusted_rand_score(y_true, y_pred))
