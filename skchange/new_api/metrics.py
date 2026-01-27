"""Metrics for changepoint detection evaluation.

This module provides two families of metrics:

1. **Changepoint metrics**: Operate on sparse changepoint indices
   - Hausdorff distance
   - F1 score with tolerance
   - Covering metric

2. **Segment metrics**: Operate on dense segment labels
   - Rand index
   - Adjusted Rand index
   - Variation of information

Design Principles
-----------------
**Per-Series Evaluation**
- All metrics operate on a SINGLE series (not collections)
- For multiple series, use list comprehension + aggregation:
  ```python
  scores = [metric(yt, yp) for yt, yp in zip(y_true_list, y_pred_list)]
  mean_score = np.mean(scores)
  ```

**Input Format**
- Primary: Segmentation dict (sparse-first principle)
- Convenience: np.ndarray (meaning documented per metric)
- Both y_true and y_pred accept same formats for symmetry

**Sklearn Compatible**
- Signature follows sklearn: metric(y_true, y_pred, ...)
- Returns float (single score per series)
- Use with make_scorer() for cross-validation
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, rand_score

from skchange.new_api.typing import Segmentation
from skchange.new_api.utils import sparse_to_dense

# ==================== Changepoint Metrics ====================


def hausdorff_metric(
    y_true: Segmentation | np.ndarray,
    y_pred: Segmentation | np.ndarray,
    max_distance: float | None = None,
) -> float:
    """Hausdorff distance between true and predicted changepoints.

    Measures the maximum distance from any true changepoint to its nearest
    predicted changepoint, and vice versa.

    **Per-series metric** - evaluates a single time series. For multiple series,
    use list comprehension:
    ```python
    scores = [hausdorff_metric(yt, yp) for yt, yp in zip(y_true, y_pred)]
    mean_score = np.mean(scores)
    ```

    Parameters
    ----------
    y_true : Segmentation | np.ndarray
        True changepoint indices for a SINGLE series.
        - Segmentation dict: extracts 'changepoints' field (preferred)
        - np.ndarray: 1D array of changepoint indices
    y_pred : Segmentation | np.ndarray
        Predicted changepoint indices for a SINGLE series.
        - Segmentation dict: extracts 'changepoints' field (preferred)
        - np.ndarray: 1D array of changepoint indices
    max_distance : float | None, default=None
        Maximum distance to clip. If None, no clipping.

    Returns
    -------
    float
        Hausdorff distance for this series. Lower is better.
        Returns 0.0 if both empty, inf (or max_distance) if only one empty.

    Examples
    --------
    >>> # Using Segmentation dict (preferred)
    >>> y_true = {
    ...     "changepoints": np.array([10, 50, 90]),
    ...     "labels": ...,
    ...     "n_samples": 200
    ... }
    >>> result = detector.predict(X)
    >>> score = hausdorff_metric(y_true, result)

    >>> # Using arrays (convenience)
    >>> y_true = np.array([10, 50, 90])
    >>> y_pred = np.array([12, 51, 88])
    >>> score = hausdorff_metric(y_true, y_pred)

    >>> # Multiple series - aggregate explicitly
    >>> y_true_list = [{"changepoints": ..., ...}, {"changepoints": ..., ...}]
    >>> y_pred_list = [detector.predict(X) for X in X_list]
    >>> scores = [hausdorff_metric(yt, yp) for yt, yp in zip(y_true_list, y_pred_list)]
    >>> mean_score = np.mean(scores)
    >>> median_score = np.median(scores)  # User controls aggregation
    """
    # Extract changepoints from Segmentation dicts
    if isinstance(y_true, dict):
        y_true = y_true["changepoints"]
    if isinstance(y_pred, dict):
        y_pred = y_pred["changepoints"]

    # Validate types
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be Segmentation dict or np.ndarray")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be Segmentation dict or np.ndarray")

    # Handle empty cases
    if len(y_true) == 0 and len(y_pred) == 0:
        return 0.0
    if len(y_true) == 0 or len(y_pred) == 0:
        return float("inf") if max_distance is None else max_distance

    # Compute Hausdorff distance
    # Distance from true to predicted
    dist_true_to_pred = np.array([np.min(np.abs(yt - y_pred)) for yt in y_true])
    # Distance from predicted to true
    dist_pred_to_true = np.array([np.min(np.abs(yp - y_true)) for yp in y_pred])

    # Hausdorff is the maximum of the two directed distances
    hausdorff = max(np.max(dist_true_to_pred), np.max(dist_pred_to_true))

    if max_distance is not None:
        hausdorff = min(hausdorff, max_distance)

    return float(hausdorff)


def f1_score(
    y_true: Segmentation | np.ndarray,
    y_pred: Segmentation | np.ndarray,
    tolerance: int = 5,
) -> float:
    """F1 score for changepoint detection with tolerance window.

    A predicted changepoint is considered a true positive if it falls within
    `tolerance` samples of a true changepoint.

    **Per-series metric** - evaluates a single time series. For multiple series,
    use list comprehension and aggregate as needed.

    Parameters
    ----------
    y_true : Segmentation | np.ndarray
        True changepoint indices for a SINGLE series.
        - Segmentation dict: extracts 'changepoints' field (preferred)
        - np.ndarray: 1D array of changepoint indices
    y_pred : Segmentation | np.ndarray
        Predicted changepoint indices for a SINGLE series.
        - Segmentation dict: extracts 'changepoints' field (preferred)
        - np.ndarray: 1D array of changepoint indices
    tolerance : int, default=5
        Maximum distance for a match to be considered correct.

    Returns
    -------
    float
        F1 score in [0, 1]. Higher is better.
        Returns 1.0 if both empty, 0.0 if only one empty.

    Examples
    --------
    >>> # Single series evaluation
    >>> y_true = {
    ...     "changepoints": np.array([10, 50, 90]),
    ...     "labels": ...,
    ...     "n_samples": 200
    ... }
    >>> result = detector.predict(X)
    >>> score = f1_score(y_true, result, tolerance=5)

    >>> # Multiple series - explicit aggregation
    >>> scores = [
    ...     f1_score(yt, yp, tolerance=5)
    ...     for yt, yp in zip(y_true_list, y_pred_list)
    ... ]
    >>> mean_f1 = np.mean(scores)
    """
    # Extract changepoints from Segmentation dicts
    if isinstance(y_true, dict):
        y_true = y_true["changepoints"]
    if isinstance(y_pred, dict):
        y_pred = y_pred["changepoints"]

    # Validate types
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be Segmentation dict or np.ndarray")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be Segmentation dict or np.ndarray")

    # Handle empty cases
    if len(y_true) == 0 and len(y_pred) == 0:
        return 1.0  # Perfect match
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0  # No match possible

    # Count true positives
    tp = 0
    matched_true = set()
    for yp in y_pred:
        # Find closest true changepoint
        distances = np.abs(y_true - yp)
        min_dist = np.min(distances)
        if min_dist <= tolerance:
            closest_idx = np.argmin(distances)
            if closest_idx not in matched_true:
                tp += 1
                matched_true.add(closest_idx)

    fp = len(y_pred) - tp
    fn = len(y_true) - tp

    # Compute F1
    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return float(f1)


# ==================== Segment Metrics ====================


def rand_index(
    y_true: Segmentation | np.ndarray,
    y_pred: Segmentation | np.ndarray,
) -> float:
    """Rand index for segment clustering quality.

    Measures the similarity between two clusterings (segmentations) by
    comparing all pairs of samples.

    **Per-series metric** - evaluates a single time series. For multiple series,
    use list comprehension and aggregate as needed.

    Parameters
    ----------
    y_true : Segmentation | np.ndarray
        True segment labels for a SINGLE series.
        - Segmentation dict: converts sparse to dense labels (preferred)
        - np.ndarray: dense labels, shape (n_samples,)
    y_pred : Segmentation | np.ndarray
        Predicted segment labels for a SINGLE series.
        - Segmentation dict: converts sparse to dense labels (preferred)
        - np.ndarray: dense labels, shape (n_samples,)

    Returns
    -------
    float
        Rand index in [0, 1]. Higher is better.

    Examples
    --------
    >>> # Using Segmentation dict (preferred)
    >>> y_true = {
    ...     "changepoints": np.array([3, 6]),
    ...     "labels": np.array([0, 1, 2]),
    ...     "n_samples": 9
    ... }
    >>> result = detector.predict(X)
    >>> score = rand_index(y_true, result)  # Auto-converts to dense

    >>> # Using dense arrays (convenience)
    >>> y_true = np.array([0,0,0,1,1,1,2,2,2])
    >>> y_pred = np.array([0,0,0,1,1,1,2,2,2])
    >>> score = rand_index(y_true, y_pred)

    >>> # Multiple series
    >>> scores = [
    ...     rand_index(yt, yp)
    ...     for yt, yp in zip(y_true_list, y_pred_list)
    ... ]
    >>> mean_ri = np.mean(scores)
    """
    # Convert Segmentation dicts to dense labels
    if isinstance(y_true, dict):
        y_true = sparse_to_dense(y_true)
    if isinstance(y_pred, dict):
        y_pred = sparse_to_dense(y_pred)

    # Validate types
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be Segmentation dict or np.ndarray")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be Segmentation dict or np.ndarray")

    # Compute Rand index using sklearn
    return float(rand_score(y_true, y_pred))


def adjusted_rand_index(
    y_true: Segmentation | np.ndarray,
    y_pred: Segmentation | np.ndarray,
) -> float:
    """Compute adjusted Rand index for segment clustering quality.

    Similar to Rand index but adjusted for chance. Value of 0 indicates
    random labeling, 1 indicates perfect agreement.

    **Per-series metric** - evaluates a single time series. For multiple series,
    use list comprehension and aggregate as needed.

    Parameters
    ----------
    y_true : Segmentation | np.ndarray
        True segment labels for a SINGLE series.
        - Segmentation dict: converts sparse to dense labels (preferred)
        - np.ndarray: dense labels, shape (n_samples,)
    y_pred : Segmentation | np.ndarray
        Predicted segment labels for a SINGLE series.
        - Segmentation dict: converts sparse to dense labels (preferred)
        - np.ndarray: dense labels, shape (n_samples,)

    Returns
    -------
    float
        Adjusted Rand index in [-1, 1]. Higher is better.
        1.0 = perfect agreement, 0.0 = random labeling.

    Examples
    --------
    >>> # Using Segmentation (preferred)
    >>> y_true = {
    ...     "changepoints": np.array([3, 6]),
    ...     "labels": ...,
    ...     "n_samples": 9
    ... }
    >>> result = detector.predict(X)
    >>> score = adjusted_rand_index(y_true, result)

    >>> # Multiple series
    >>> scores = [
    ...     adjusted_rand_index(yt, yp)
    ...     for yt, yp in zip(y_true_list, y_pred_list)
    ... ]
    >>> mean_ari = np.mean(scores)
    """
    # Convert Segmentation dicts to dense labels
    if isinstance(y_true, dict):
        y_true = sparse_to_dense(y_true)
    if isinstance(y_pred, dict):
        y_pred = sparse_to_dense(y_pred)

    # Validate types
    if not isinstance(y_true, np.ndarray):
        raise TypeError("y_true must be Segmentation dict or np.ndarray")
    if not isinstance(y_pred, np.ndarray):
        raise TypeError("y_pred must be Segmentation dict or np.ndarray")

    # Compute ARI using sklearn
    return float(adjusted_rand_score(y_true, y_pred))


# ==================== Scorer Utilities ====================


def make_changepoint_scorer(metric_func, needs_X=False):
    """Create a scorer compatible with sklearn cross-validation.

    sklearn's cross_val_score and GridSearchCV expect scorers with signature
    scorer(estimator, X, y). This function wraps changepoint metrics to
    match that interface.

    Parameters
    ----------
    metric_func : callable
        Metric function with signature metric(y_true, y_pred, X=None).
        Should return a scalar score.
    needs_X : bool, default=False
        Whether the metric requires X data.
        - If False: calls metric(y_true, y_pred)
        - If True: calls metric(y_true, y_pred, X)

    Returns
    -------
    scorer : callable
        Scorer function with signature scorer(estimator, X, y) compatible
        with sklearn's cross_val_score, GridSearchCV, etc.

    Examples
    --------
    >>> from sklearn.model_selection import cross_val_score
    >>> from skchange.metrics import hausdorff_metric, make_changepoint_scorer
    >>>
    >>> # Metric that doesn't need X
    >>> scorer = make_changepoint_scorer(hausdorff_metric)
    >>> scores = cross_val_score(detector, X_list, y_list, scoring=scorer)
    >>>
    >>> # Metric that needs X
    >>> scorer = make_changepoint_scorer(segment_mse, needs_X=True)
    >>> scores = cross_val_score(detector, X_list, y_list, scoring=scorer)
    """

    def scorer(estimator, X, y):
        """Scorer wrapper for sklearn compatibility."""
        y_pred = estimator.predict(X)
        if needs_X:
            return metric_func(y, y_pred, X)
        else:
            return metric_func(y, y_pred)

    return scorer
