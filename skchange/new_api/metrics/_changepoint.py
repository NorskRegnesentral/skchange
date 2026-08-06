"""Changepoint-based evaluation metrics.

Functions in this module take **changepoint index arrays** as their native input
— the sorted integer arrays returned by ``predict()``.

Every function accepts ``(changepoints_true, changepoints_pred, n_samples, ...)``
so metrics are interchangeable in a strategy pattern. Metrics that intrinsically
depend on the series length (``rand_index``, ``adjusted_rand_index``) make
``n_samples`` a required argument. Metrics that are length-invariant (Hausdorff,
precision, recall, F1) accept ``n_samples`` for signature uniformity but ignore
it, so it defaults to ``None``.
"""

import numbers

import numpy as np
from sklearn.metrics import adjusted_rand_score as _adjusted_rand_score
from sklearn.metrics import rand_score as _rand_score

from skchange.new_api.types import ArrayLike
from skchange.new_api.utils._param_validation import Interval, validate_params
from skchange.new_api.utils.segmentation import changepoints_to_labels


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "max_distance": [Interval(numbers.Real, 0, None, closed="left"), None],
    },
    prefer_skip_nested_validation=True,
)
def hausdorff_metric(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int | None = None,
    *,
    max_distance: float | None = None,
) -> float:
    """Compute the Hausdorff distance between two changepoint sets.

    Measures the maximum distance from any true changepoint to its nearest
    predicted changepoint, and vice versa. Lower is better.

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int | None, default=None
        Accepted for signature uniformity across changepoint metrics; unused.
    max_distance : float | None, default=None
        Cap on the returned distance. If None, no cap is applied.

    Returns
    -------
    float
        Hausdorff distance. Returns 0.0 if both have no changepoints,
        inf (or max_distance) if exactly one has no changepoints.

    Examples
    --------
    >>> hausdorff_metric([10, 20], [12, 20])
    2.0
    """
    del n_samples  # accepted for signature uniformity; unused
    cp_true = np.asarray(changepoints_true)
    cp_pred = np.asarray(changepoints_pred)

    if len(cp_true) == 0 and len(cp_pred) == 0:
        return 0.0
    if len(cp_true) == 0 or len(cp_pred) == 0:
        return float("inf") if max_distance is None else float(max_distance)

    dist_true_to_pred = np.array([np.min(np.abs(yt - cp_pred)) for yt in cp_true])
    dist_pred_to_true = np.array([np.min(np.abs(yp - cp_true)) for yp in cp_pred])
    hausdorff = float(max(np.max(dist_true_to_pred), np.max(dist_pred_to_true)))

    if max_distance is not None:
        hausdorff = min(hausdorff, float(max_distance))
    return hausdorff


def _count_tp(
    cp_true: np.ndarray,
    cp_pred: np.ndarray,
    tolerance: int,
) -> int:
    """Count true positives via greedy tolerance matching."""
    matched_true: set[int] = set()
    tp = 0
    for yp in cp_pred:
        distances = np.abs(cp_true - yp)
        min_dist_idx = int(np.argmin(distances))
        if distances[min_dist_idx] <= tolerance and min_dist_idx not in matched_true:
            tp += 1
            matched_true.add(min_dist_idx)
    return tp


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "tolerance": [Interval(numbers.Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def changepoint_precision(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int | None = None,
    *,
    tolerance: int = 5,
) -> float:
    """Compute detection precision for changepoints with a tolerance window.

    The fraction of predicted changepoints that match a true changepoint within
    ``tolerance`` samples (greedy matching). Higher is better.

    Returns 1.0 when there are no predicted changepoints (no false alarms).

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int | None, default=None
        Accepted for signature uniformity across changepoint metrics; unused.
    tolerance : int, default=5
        Maximum sample distance for a match to count as correct.

    Returns
    -------
    float
        Precision in [0, 1].

    Examples
    --------
    >>> changepoint_precision([10, 20], [12, 20], tolerance=5)
    1.0
    """
    del n_samples  # accepted for signature uniformity; unused
    cp_true = np.asarray(changepoints_true)
    cp_pred = np.asarray(changepoints_pred)

    if len(cp_pred) == 0:
        return 1.0
    if len(cp_true) == 0:
        return 0.0

    tp = _count_tp(cp_true, cp_pred, tolerance)
    return float(tp / len(cp_pred))


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "tolerance": [Interval(numbers.Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def changepoint_recall(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int | None = None,
    *,
    tolerance: int = 5,
) -> float:
    """Compute detection recall for changepoints with a tolerance window.

    The fraction of true changepoints that are matched by a predicted
    changepoint within ``tolerance`` samples (greedy matching). Higher is better.

    Returns 1.0 when there are no true changepoints (nothing to miss).

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int | None, default=None
        Accepted for signature uniformity across changepoint metrics; unused.
    tolerance : int, default=5
        Maximum sample distance for a match to count as correct.

    Returns
    -------
    float
        Recall in [0, 1].

    Examples
    --------
    >>> changepoint_recall([10, 20], [12, 20], tolerance=5)
    1.0
    """
    del n_samples  # accepted for signature uniformity; unused
    cp_true = np.asarray(changepoints_true)
    cp_pred = np.asarray(changepoints_pred)

    if len(cp_true) == 0:
        return 1.0
    if len(cp_pred) == 0:
        return 0.0

    tp = _count_tp(cp_true, cp_pred, tolerance)
    return float(tp / len(cp_true))


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left"), None],
        "tolerance": [Interval(numbers.Integral, 0, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def changepoint_f1_score(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int | None = None,
    *,
    tolerance: int = 5,
) -> float:
    """Compute the F1 score for changepoint detection with a tolerance window.

    Harmonic mean of ``changepoint_precision`` and ``changepoint_recall``.
    A predicted changepoint is a true positive if it falls within ``tolerance``
    samples of an unmatched true changepoint (greedy matching).

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int | None, default=None
        Accepted for signature uniformity across changepoint metrics; unused.
    tolerance : int, default=5
        Maximum sample distance for a match to count as correct.

    Returns
    -------
    float
        F1 score in [0, 1]. Higher is better.
        Returns 1.0 if both have no changepoints, 0.0 if only one has none.

    Examples
    --------
    >>> changepoint_f1_score([10, 20], [12, 20], tolerance=5)
    1.0
    """
    del n_samples  # accepted for signature uniformity; unused
    precision = changepoint_precision(
        changepoints_true, changepoints_pred, tolerance=tolerance
    )
    recall = changepoint_recall(
        changepoints_true, changepoints_pred, tolerance=tolerance
    )
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def rand_index(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int,
) -> float:
    """Compute the Rand index between two changepoint-defined segmentations.

    Inflates both changepoint arrays to per-sample segment labels via
    :func:`skchange.new_api.utils.segmentation.changepoints_to_labels` and
    delegates to :func:`sklearn.metrics.rand_score`. Higher is better.

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int
        Length of the underlying time series. The Rand index depends on segment
        lengths, which cannot be inferred from changepoint indices alone.

    Returns
    -------
    float
        Rand index in [0, 1]. Higher is better.

    Examples
    --------
    >>> rand_index([50], [50], n_samples=100)
    1.0
    """
    labels_true = changepoints_to_labels(
        np.asarray(changepoints_true), n_samples=n_samples
    )
    labels_pred = changepoints_to_labels(
        np.asarray(changepoints_pred), n_samples=n_samples
    )
    return float(_rand_score(labels_true, labels_pred))


@validate_params(
    {
        "changepoints_true": ["array-like"],
        "changepoints_pred": ["array-like"],
        "n_samples": [Interval(numbers.Integral, 1, None, closed="left")],
    },
    prefer_skip_nested_validation=True,
)
def adjusted_rand_index(
    changepoints_true: ArrayLike,
    changepoints_pred: ArrayLike,
    n_samples: int,
) -> float:
    """Compute the adjusted Rand index between two changepoint-defined segmentations.

    Similar to :func:`rand_index` but adjusted for chance. Inflates both
    changepoint arrays to per-sample labels by
    :func:`skchange.new_api.utils.segmentation.changepoints_to_labels`
    and delegates to
    :func:`sklearn.metrics.adjusted_rand_score` for the computation.

    Parameters
    ----------
    changepoints_true : array-like of shape (n_changepoints_true,)
        True changepoint indices, as returned by ``predict()``.
    changepoints_pred : array-like of shape (n_changepoints_pred,)
        Predicted changepoint indices, as returned by ``predict()``.
    n_samples : int
        Length of the underlying time series. The adjusted Rand index depends
        on segment lengths, which cannot be inferred from changepoint indices
        alone.

    Returns
    -------
    float
        Adjusted Rand index in [-1, 1]. Higher is better.
        1.0 = perfect agreement, ~0.0 = random labeling.

    Examples
    --------
    >>> adjusted_rand_index([50], [50], n_samples=100)
    1.0
    """
    labels_true = changepoints_to_labels(
        np.asarray(changepoints_true), n_samples=n_samples
    )
    labels_pred = changepoints_to_labels(
        np.asarray(changepoints_pred), n_samples=n_samples
    )
    return float(_adjusted_rand_score(labels_true, labels_pred))
