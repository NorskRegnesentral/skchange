"""Metrics for changepoint detection evaluation.

All metrics follow a uniform interface: they take per-sample segment label
arrays as ``y_true`` and ``y_pred``, matching the output of
``BaseChangeDetector.predict()``. This makes all metrics interchangeable.

If you have changepoint indices (e.g. from ``predict_changepoints()`` or
external annotations), convert them first::

    from skchange.new_api.utils import changepoints_to_labels
    y_true = changepoints_to_labels(true_changepoints, n_samples=len(X))
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score

from skchange.new_api.utils._conversion import labels_to_changepoints


def hausdorff_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    max_distance: float | None = None,
) -> float:
    """Compute the Hausdorff distance between the changepoint sets of two segmentations.

    Accepts dense per-sample segment label arrays (output of ``predict()``).
    Changepoint indices are extracted internally.

    Measures the maximum distance from any true changepoint to its nearest
    predicted changepoint, and vice versa. Lower is better.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels for a single series.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels for a single series.
    max_distance : float | None, default=None
        Cap on the returned distance. If None, no cap is applied.

    Returns
    -------
    float
        Hausdorff distance. Returns 0.0 if both have no changepoints,
        inf (or max_distance) if exactly one has no changepoints.

    Examples
    --------
    >>> y_true = np.array([0]*10 + [1]*10 + [2]*10)
    >>> y_pred = np.array([0]*12 + [1]*8 + [2]*10)
    >>> hausdorff_metric(y_true, y_pred)
    2.0
    """
    cp_true = labels_to_changepoints(np.asarray(y_true))
    cp_pred = labels_to_changepoints(np.asarray(y_pred))

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


def changepoint_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tolerance: int = 5,
) -> float:
    """Compute the F1 score for changepoint detection with a tolerance window.

    Accepts dense per-sample segment label arrays (output of ``predict()``).
    Changepoint indices are extracted internally.

    A predicted changepoint is a true positive if it falls within ``tolerance``
    samples of an unmatched true changepoint (greedy matching).

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels for a single series.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels for a single series.
    tolerance : int, default=5
        Maximum sample distance for a match to count as correct.

    Returns
    -------
    float
        F1 score in [0, 1]. Higher is better.
        Returns 1.0 if both have no changepoints, 0.0 if only one has none.

    Examples
    --------
    >>> y_true = np.array([0]*10 + [1]*10 + [2]*10)
    >>> y_pred = np.array([0]*12 + [1]*8 + [2]*10)
    >>> changepoint_f1_score(y_true, y_pred, tolerance=5)
    1.0
    """
    cp_true = labels_to_changepoints(np.asarray(y_true))
    cp_pred = labels_to_changepoints(np.asarray(y_pred))

    if len(cp_true) == 0 and len(cp_pred) == 0:
        return 1.0
    if len(cp_true) == 0 or len(cp_pred) == 0:
        return 0.0

    tp = 0
    matched_true = set()
    for yp in cp_pred:
        distances = np.abs(cp_true - yp)
        min_dist_idx = int(np.argmin(distances))
        if distances[min_dist_idx] <= tolerance and min_dist_idx not in matched_true:
            tp += 1
            matched_true.add(min_dist_idx)

    fp = len(cp_pred) - tp
    fn = len(cp_true) - tp
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return float(2 * precision * recall / (precision + recall))


def rand_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the Rand index for the segmentation.

    Measures the similarity between two segmentations by comparing all pairs of
    samples. Wraps ``sklearn.metrics.rand_score``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels for a single series.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels for a single series.

    Returns
    -------
    float
        Rand index in [0, 1]. Higher is better.

    Examples
    --------
    >>> y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> rand_index(y_true, y_pred)
    1.0
    """
    return float(rand_score(y_true, y_pred))


def adjusted_rand_index(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute the adjusted Rand index for the segmentation.

    Similar to Rand index but adjusted for chance. Wraps
    ``sklearn.metrics.adjusted_rand_score``.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
        True segment labels for a single series.
    y_pred : np.ndarray of shape (n_samples,)
        Predicted segment labels for a single series.

    Returns
    -------
    float
        Adjusted Rand index in [-1, 1]. Higher is better.
        1.0 = perfect agreement, ~0.0 = random labeling.

    Examples
    --------
    >>> y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    >>> adjusted_rand_index(y_true, y_pred)
    1.0
    """
    return float(adjusted_rand_score(y_true, y_pred))
