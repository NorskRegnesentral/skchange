import numpy as np


def to_labels(
    changepoints: np.ndarray,
    n_samples: int,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Convert changepoint indices to per-sample segment labels.

    Parameters
    ----------
    changepoints : np.ndarray
        Changepoint indices, shape (n_changepoints,).
    n_samples : int
        Number of samples in the time series.
    labels : np.ndarray | None, default=None
        Segment labels, shape (n_changepoints + 1,).
        If None, auto-generates [0, 1, 2, ...].

    Returns
    -------
    np.ndarray
        Segment labels, shape (n_samples,). Each sample assigned its segment label.

    Examples
    --------
    >>> changepoints = np.array([50, 100])
    >>> labels = to_labels(changepoints, n_samples=150)
    >>> labels.shape
    (150,)
    >>> np.unique(labels)
    array([0, 1, 2])
    """
    changepoints = np.asarray(changepoints, dtype=int)

    if labels is None:
        labels = np.arange(len(changepoints) + 1, dtype=int)
    else:
        labels = np.asarray(labels, dtype=int)

    dense_labels = np.zeros(n_samples, dtype=int)

    if len(changepoints) > 0:
        boundaries = np.concatenate([[0], changepoints, [n_samples]])
        for seg_id in range(len(boundaries) - 1):
            start = boundaries[seg_id]
            end = boundaries[seg_id + 1]
            dense_labels[start:end] = labels[seg_id]

    return dense_labels
