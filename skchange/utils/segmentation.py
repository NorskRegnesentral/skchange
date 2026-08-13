"""Utility functions for converting between different change detector output formats."""

import numpy as np


def changepoints_to_labels(
    changepoints: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Convert changepoint indices to per-sample segment labels.

    Parameters
    ----------
    changepoints : np.ndarray
        Changepoint indices, shape (n_changepoints,).
    n_samples : int
        Number of samples in the time series.

    Returns
    -------
    np.ndarray
        Segment labels, shape (n_samples,). Each sample assigned its segment label.

    Examples
    --------
    >>> changepoints = np.array([50, 100])
    >>> labels = changepoints_to_labels(changepoints, n_samples=150)
    >>> labels.shape
    (150,)
    >>> np.unique(labels)
    array([0, 1, 2])
    """
    changepoints = np.asarray(changepoints, dtype=int)
    labels = np.arange(len(changepoints) + 1, dtype=int)
    dense_labels = np.zeros(n_samples, dtype=int)

    if len(changepoints) > 0:
        boundaries = np.concatenate([[0], changepoints, [n_samples]])
        for seg_id in range(len(boundaries) - 1):
            start = boundaries[seg_id]
            end = boundaries[seg_id + 1]
            dense_labels[start:end] = labels[seg_id]

    return dense_labels
