"""Segment-anomaly-based evaluation metrics.

Functions in this module take **interval arrays** as their native input —
arrays of shape ``(n_anomalies, 2)`` where each row is a ``[start, end)``
index pair, as returned by ``predict_segment_anomalies()``.
"""

import numpy as np


def _count_tp(it: np.ndarray, ip: np.ndarray) -> int:
    """Count true positives via greedy overlap matching."""
    matched_true: set[int] = set()
    tp = 0
    for start_pred, end_pred in ip:
        for j, (start_true, end_true) in enumerate(it):
            if j in matched_true:
                continue
            if max(start_pred, start_true) < min(end_pred, end_true):
                tp += 1
                matched_true.add(j)
                break
    return tp


def segment_anomaly_precision(
    intervals_true: np.ndarray,
    intervals_pred: np.ndarray,
) -> float:
    """Compute detection precision for segment anomalies.

    The fraction of predicted anomalous intervals that overlap with a true
    anomalous interval (greedy matching). Higher is better.

    Returns 1.0 when there are no predicted intervals (no false alarms).

    Parameters
    ----------
    intervals_true : np.ndarray of shape (n_true, 2)
        True anomalous intervals, as returned by ``predict_segment_anomalies()``.
        Each row is a ``[start, end)`` index pair.
    intervals_pred : np.ndarray of shape (n_pred, 2)
        Predicted anomalous intervals, as returned by
        ``predict_segment_anomalies()``. Each row is a ``[start, end)`` index pair.

    Returns
    -------
    float
        Precision in [0, 1].

    Examples
    --------
    >>> segment_anomaly_precision(np.array([[10, 20]]), np.array([[12, 22]]))
    1.0
    """
    it = np.asarray(intervals_true)
    ip = np.asarray(intervals_pred)

    if len(ip) == 0:
        return 1.0
    if len(it) == 0:
        return 0.0

    tp = _count_tp(it, ip)
    return float(tp / len(ip))


def segment_anomaly_recall(
    intervals_true: np.ndarray,
    intervals_pred: np.ndarray,
) -> float:
    """Compute detection recall for segment anomalies.

    The fraction of true anomalous intervals that are matched by a predicted
    anomalous interval (greedy matching). Higher is better.

    Returns 1.0 when there are no true intervals (nothing to miss).

    Parameters
    ----------
    intervals_true : np.ndarray of shape (n_true, 2)
        True anomalous intervals, as returned by ``predict_segment_anomalies()``.
        Each row is a ``[start, end)`` index pair.
    intervals_pred : np.ndarray of shape (n_pred, 2)
        Predicted anomalous intervals, as returned by
        ``predict_segment_anomalies()``. Each row is a ``[start, end)`` index pair.

    Returns
    -------
    float
        Recall in [0, 1].

    Examples
    --------
    >>> segment_anomaly_recall(np.array([[10, 20]]), np.array([[12, 22]]))
    1.0
    """
    it = np.asarray(intervals_true)
    ip = np.asarray(intervals_pred)

    if len(it) == 0:
        return 1.0
    if len(ip) == 0:
        return 0.0

    tp = _count_tp(it, ip)
    return float(tp / len(it))


def segment_anomaly_f1_score(
    intervals_true: np.ndarray,
    intervals_pred: np.ndarray,
) -> float:
    """Compute the F1 score for segment anomaly detection.

    Harmonic mean of ``segment_anomaly_precision`` and
    ``segment_anomaly_recall``. A predicted anomalous interval is a true
    positive if it overlaps with an unmatched true anomalous interval (greedy
    matching). Two intervals overlap when
    ``max(start_true, start_pred) < min(end_true, end_pred)``.

    Parameters
    ----------
    intervals_true : np.ndarray of shape (n_true, 2)
        True anomalous intervals, as returned by ``predict_segment_anomalies()``.
        Each row is a ``[start, end)`` index pair.
    intervals_pred : np.ndarray of shape (n_pred, 2)
        Predicted anomalous intervals, as returned by
        ``predict_segment_anomalies()``.  Each row is a ``[start, end)`` index
        pair.

    Returns
    -------
    float
        F1 score in [0, 1]. Higher is better.
        Returns 1.0 if both have no anomalies, 0.0 if only one has none.

    Examples
    --------
    >>> intervals_true = np.array([[10, 20], [50, 60]])
    >>> intervals_pred = np.array([[12, 22], [50, 60]])
    >>> segment_anomaly_f1_score(intervals_true, intervals_pred)
    1.0
    """
    precision = segment_anomaly_precision(intervals_true, intervals_pred)
    recall = segment_anomaly_recall(intervals_true, intervals_pred)
    if precision + recall == 0.0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))
