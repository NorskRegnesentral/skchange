"""Utility functions for the new skchange API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.utils._tags import Tags

from skchange.new_api.typing import Segmentation


@dataclass(slots=True)
class ChangeDetectorTags:
    """Tags specific to change detection estimators.

    Attributes
    ----------
    multivariate : bool, default=True
        Whether the detector can handle multivariate time series (n_features > 1).
    supervised : bool, default=False
        Whether the detector supports supervised learning (requires y labels).
    variable_identification : bool, default=False
        Whether the detector can identify which variables are affected
        at each changepoint.
    integer_input_only : bool, default=False
        Whether the detector requires integer-valued input data (e.g., for count data).
    """

    multivariate: bool = True
    supervised: bool = False
    variable_identification: bool = False
    integer_input_only: bool = False


@dataclass
class SkchangeTags(Tags):
    """Extended tags for skchange estimators.

    Extends sklearn's base Tags with change detection specific tag group.

    Attributes
    ----------
    change_detector_tags : ChangeDetectorTags
        Change detection specific tags.
    """

    change_detector_tags: ChangeDetectorTags = field(default_factory=ChangeDetectorTags)


def make_segmentation(
    changepoints: np.ndarray,
    n_samples: int,
    labels: np.ndarray | None = None,
    n_features: int | None = None,
    changed_features: list[np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> Segmentation:
    """Create a Segmentation dict with clean syntax.

    This helper function mimics dataclass-style construction while returning
    a plain dict. Auto-generates labels if not provided.

    Required Parameters (Always Included in Output)
    -----------------------------------------------
    changepoints : np.ndarray
        Changepoint indices, shape (n_changepoints,). REQUIRED.
    n_samples : int
        Number of samples in the time series. REQUIRED.
    labels : np.ndarray | None, default=None
        Segment labels, shape (n_changepoints + 1,). REQUIRED in output.
        If None, auto-generates default [0, 1, 2, ...] labels.

    Optional Parameters (Included Only If Provided)
    ------------------------------------------------
    n_features : int | None, default=None
        Number of features/channels. Only added if not None.
    changed_features : list[np.ndarray] | None, default=None
        Features affected at each changepoint. Only added if not None.
    meta : dict[str, Any] | None, default=None
        Additional metadata. Only added if not None.

    Returns
    -------
    Segmentation
        A typed dict with the specified fields

    Examples
    --------
    >>> # Basic usage
    >>> X = np.random.randn(200, 3)
    >>> result = make_segmentation(
    ...     changepoints=np.array([50, 100]),
    ...     n_samples=X.shape[0],
    ...     labels=np.array([0, 1, 2]),
    ... )
    >>> result["changepoints"]
    array([50, 100])
    >>> result["labels"]
    array([0, 1, 2])

    >>> # With optional n_features
    >>> result = make_segmentation(
    ...     changepoints=np.array([50, 100]),
    ...     n_samples=200,
    ...     labels=np.array([0, 1, 0]),  # Return to state 0
    ...     n_features=3,
    ... )
    """
    # Auto-generate labels if not provided
    if labels is None:
        n_changepoints = len(changepoints) if changepoints is not None else 0
        labels = np.arange(n_changepoints + 1)

    # Build result dict
    result: Segmentation = {
        "changepoints": changepoints,
        "labels": labels,
        "n_samples": n_samples,
    }

    # Add optional fields only if provided
    if n_features is not None:
        result["n_features"] = n_features

    if changed_features is not None:
        result["changed_features"] = changed_features

    if meta is not None:
        result["meta"] = meta

    return result


def validate_data(X: Any, y: Any = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Validate input data, ensuring X is 2D and optionally validating y.

    Similar to scikit-learn's check_array and check_X_y pattern.

    Parameters
    ----------
    X : ArrayLike
        Input data.
    y : ArrayLike, optional
        Target values. If provided, will be validated and returned.

    Returns
    -------
    np.ndarray or tuple[np.ndarray, np.ndarray]
        If y is None: returns validated X as 2D array.
        If y is provided: returns (X, y) tuple with both validated.

    Raises
    ------
    ValueError
        If X cannot be converted to 2D or has wrong number of dimensions.
        If y is provided and has incompatible length with X.

    Examples
    --------
    >>> # Validate X only - univariate series
    >>> X = np.array([1, 2, 3, 4, 5])
    >>> X_val = validate_data(X)
    >>> X_val.shape
    (5, 1)

    >>> # Validate X only - multivariate series
    >>> X = np.random.randn(100, 3)
    >>> X_val = validate_data(X)
    >>> X_val.shape
    (100, 3)

    >>> # Validate both X and y
    >>> X = np.random.randn(100, 3)
    >>> y = np.array([0, 1] * 50)
    >>> X_val, y_val = validate_data(X, y)
    >>> X_val.shape, y_val.shape
    ((100, 3), (100,))
    """
    X = np.asarray(X)

    if X.ndim == 1:
        # Univariate series given as 1D - reshape to (n, 1)
        X = X.reshape(-1, 1)
    elif X.ndim == 2:
        # Already 2D - good
        pass
    else:
        raise ValueError(
            f"Expected 1D or 2D array, got {X.ndim}D array with shape {X.shape}. "
            f"For single series use shape (n_timepoints, n_channels). "
            f"For multiple series pass a list of 2D arrays."
        )

    if y is None:
        return X

    # Validate y
    y = np.asarray(y)
    if y.ndim > 1:
        y = y.ravel()

    if len(y) != len(X):
        raise ValueError(
            f"X and y must have the same length. Got len(X)={len(X)}, len(y)={len(y)}"
        )

    return X, y


def sparse_to_dense(result: Segmentation) -> np.ndarray:
    """Convert sparse segmentation to dense segment labels.

    Parameters
    ----------
    result : Segmentation
        Dict with required fields:

        - "changepoints": np.ndarray of changepoint indices
        - "labels": np.ndarray of segment labels
        - "n_samples": int, number of samples

    Returns
    -------
    np.ndarray
        Dense labels, shape (n_samples,). Each segment gets a unique integer label.

    Examples
    --------
    >>> result = {
    ...     "changepoints": np.array([50, 100]),
    ...     "n_samples": 150,
    ...     "n_features": 3,
    ...     "labels": np.array([0, 1, 2]),
    ... }
    >>> dense_labels = sparse_to_dense(result)
    >>> dense_labels.shape
    (150,)
    >>> np.unique(dense_labels)
    array([0, 1, 2])
    """
    n_samples = result["n_samples"]
    dense_labels = np.zeros(n_samples, dtype=int)

    if result["changepoints"] is not None:
        changepoints = np.asarray(result["changepoints"])
        if len(changepoints) > 0:
            # Create segment labels
            changepoints = np.concatenate([[0], changepoints, [n_samples]])
            for seg_id in range(len(changepoints) - 1):
                start = changepoints[seg_id]
                end = changepoints[seg_id + 1]
                dense_labels[start:end] = seg_id

    return dense_labels


def dense_to_sparse(labels: np.ndarray, n_samples: int | None = None) -> Segmentation:
    """Convert dense segment labels to sparse segmentation.

    Parameters
    ----------
    labels : np.ndarray
        Dense segment labels, shape (n_samples,). Each segment should have
        a consistent label.
    n_samples : int | None, optional
        Number of samples. If None, inferred from len(labels).

    Returns
    -------
    Segmentation
        Dict with required fields:

        - "changepoints": np.ndarray, extracted from label transitions
        - "labels": np.ndarray, segment labels
        - "n_samples": int, number of samples

    Examples
    --------
    >>> # Dense labels with 3 segments
    >>> labels = np.array([0,0,0,1,1,1,2,2,2])
    >>> result = dense_to_sparse(labels)
    >>> result["changepoints"]
    array([3, 6])
    >>> result["labels"]
    array([0, 1, 2])
    >>> result["n_samples"]
    9

    >>> # Labels with repeated segments (same label returns)
    >>> labels = np.array([0,0,1,1,2,2,0,0])
    >>> result = dense_to_sparse(labels)
    >>> result["changepoints"]
    array([2, 4, 6])
    >>> result["labels"]
    array([0, 1, 2, 0])
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1D array. Got shape {labels.shape}")

    if n_samples is None:
        n_samples = len(labels)
    elif n_samples != len(labels):
        raise ValueError(
            f"n_samples={n_samples} does not match len(labels)={len(labels)}"
        )

    # Find changepoints where label changes
    if len(labels) == 0:
        changepoints = np.array([], dtype=int)
        segment_labels = np.array([], dtype=int)
    else:
        # Indices where labels change
        changes = np.where(np.diff(labels) != 0)[0] + 1
        changepoints = changes.astype(int)

        # Extract segment labels
        segment_starts = np.concatenate([[0], changepoints])
        segment_labels = labels[segment_starts]

    return make_segmentation(
        changepoints=changepoints,
        labels=segment_labels,
        n_samples=n_samples,
    )
