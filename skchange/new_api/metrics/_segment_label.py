"""Segment-label-based evaluation metrics.

Functions in this module take **dense per-sample label arrays** as their native
input — the integer arrays returned by ``predict()``.
"""

from sklearn.metrics import adjusted_rand_score as _adjusted_rand_score
from sklearn.metrics import rand_score as _rand_score
from sklearn.utils._param_validation import validate_params

from skchange.new_api.typing import ArrayLike


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def rand_index(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Compute the Rand index for two segmentations.

    Measures the similarity between two segmentations by comparing all pairs of
    samples. Wraps ``sklearn.metrics.rand_score``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True segment labels, as returned by ``predict()``.
    y_pred : array-like of shape (n_samples,)
        Predicted segment labels, as returned by ``predict()``.

    Returns
    -------
    float
        Rand index in [0, 1]. Higher is better.

    Examples
    --------
    >>> rand_index([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    1.0
    """
    return float(_rand_score(y_true, y_pred))


@validate_params(
    {
        "y_true": ["array-like"],
        "y_pred": ["array-like"],
    },
    prefer_skip_nested_validation=True,
)
def adjusted_rand_index(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> float:
    """Compute the adjusted Rand index for two segmentations.

    Similar to Rand index but adjusted for chance. Wraps
    ``sklearn.metrics.adjusted_rand_score``.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True segment labels, as returned by ``predict()``.
    y_pred : array-like of shape (n_samples,)
        Predicted segment labels, as returned by ``predict()``.

    Returns
    -------
    float
        Adjusted Rand index in [-1, 1]. Higher is better.
        1.0 = perfect agreement, ~0.0 = random labeling.

    Examples
    --------
    >>> adjusted_rand_index([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    1.0
    """
    return float(_adjusted_rand_score(y_true, y_pred))
