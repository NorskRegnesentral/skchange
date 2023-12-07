"""Utility functions for change detection."""

import numpy as np
import pandas as pd


def changepoints_to_labels(changepoints: list) -> np.ndarray:
    """Convert a list of changepoints to a list of labels.

    Parameters
    ----------
    changepoints : list
        List of changepoint indices.
    n: int
        Sample size.

    Returns
    -------
    labels : np.ndarray
        1D array of labels: 0 for the first segment, 1 for the second, etc.
    """
    changepoints = [-1] + changepoints
    n = changepoints[-1] + 1  # The last changepoint is always the last data index
    labels = np.zeros(n)
    for i in range(len(changepoints) - 1):
        labels[changepoints[i] + 1 : changepoints[i + 1] + 1] = i
    return labels


def format_changepoint_output(
    fmt: str,
    labels: str,
    changepoints: list,
    X_index: pd.Index,
    scores: np.ndarray = None,
) -> pd.Series:
    """Format the predict method output of change detectors.

    Parameters
    ----------
    fmt : str
        Format of the output. Either "sparse" or "dense".
    labels : str
        Labels of the output. Either "indicator", "score" or "int_label".
    changepoints : list
        List of changepoint indices.
    X_index : pd.Index
        Index of the input data.
    scores : np.ndarray, optional (default=None)
        Array of scores.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
    if labels == "indicator":
        out = pd.Series(False, index=X_index)
        out.iloc[changepoints] = True
    elif labels == "score":
        out = pd.Series(scores, index=X_index)
    elif labels == "int_label":
        out = changepoints_to_labels(changepoints)
        out = pd.Series(out, index=X_index)

    if fmt == "sparse":
        out = out.iloc[changepoints]

    return out
