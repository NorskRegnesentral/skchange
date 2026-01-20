"""Utility functions for the new skchange API."""

from __future__ import annotations

from typing import Any

import numpy as np

from .typing import ChangeDetectionResult


def make_change_detection_result(
    indices: np.ndarray,
    n_samples: int,
    n_features: int,
    segment_labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    affected_variables: list[np.ndarray] | None = None,
    meta: dict[str, Any] | None = None,
) -> ChangeDetectionResult:
    """Create a ChangeDetectionResult dict with clean syntax.

    This helper function mimics dataclass-style construction while returning
    a plain dict. Provides sensible defaults for optional fields.

    All array outputs are numpy arrays for consistency (following sklearn convention).

    Parameters
    ----------
    indices : np.ndarray
        Changepoint indices, shape (n_changepoints,)
    n_samples : int
        Number of samples in the time series
    n_features : int
        Number of features in the time series
    segment_labels : np.ndarray | None, optional
        Segment labels, shape (n_changepoints + 1,).
        If None, generates default [0, 1, 2, ...] labels.
    scores : np.ndarray | None, optional
        Scores at each changepoint
    affected_variables : list[np.ndarray] | None, optional
        Variables affected at each changepoint (list of numpy arrays)
    meta : dict[str, Any] | None, optional
        Additional metadata

    Returns
    -------
    ChangeDetectionResult
        A typed dict with the specified fields

    Examples
    --------
    >>> # Basic usage
    >>> X = np.random.randn(200, 3)
    >>> result = make_change_detection_result(
    ...     indices=np.array([50, 100]),
    ...     n_samples=X.shape[0],
    ...     n_features=X.shape[1],
    ... )
    >>> result["indices"]
    array([50, 100])
    >>> result["segment_labels"]  # Auto-generated
    array([0, 1, 2])

    >>> # With custom segment labels
    >>> result = make_change_detection_result(
    ...     indices=np.array([50, 100]),
    ...     n_samples=200,
    ...     n_features=3,
    ...     segment_labels=np.array([0, 1, 0]),  # Return to state 0
    ... )
    """
    # Auto-generate segment labels if not provided
    if segment_labels is None:
        n_changepoints = len(indices) if indices is not None else 0
        segment_labels = np.arange(n_changepoints + 1)

    # Build result dict
    result: ChangeDetectionResult = {
        "indices": indices,
        "segment_labels": segment_labels,
        "n_samples": n_samples,
        "n_features": n_features,
    }

    # Add optional fields only if provided
    if scores is not None:
        result["scores"] = scores
    if affected_variables is not None:
        result["affected_variables"] = affected_variables
    if meta is not None:
        result["meta"] = meta
    else:
        result["meta"] = {}  # Default empty dict

    return result
