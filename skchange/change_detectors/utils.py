"""Utility functions for change detection."""

from typing import Union

import numpy as np
import pandas as pd


def changepoints_to_labels(changepoints: list, n) -> np.ndarray:
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
    changepoints = [-1] + changepoints + [n - 1]
    labels = np.zeros(n)
    for i in range(len(changepoints) - 1):
        labels[changepoints[i] + 1 : changepoints[i + 1] + 1] = i
    return labels


def format_changepoint_output(
    fmt: str,
    labels: str,
    changepoints: list,
    X_index: pd.Index,
    scores: Union[pd.Series, pd.DataFrame] = None,
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
    scores : pd.Series or pd.DataFrame, optional (default=None)
        Series or DataFrame of scores. If Series, it must be named 'score', and if
        DataFrame, it must have a column named 'score'.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
    if fmt == "sparse" and labels in ["int_label", "indicator"]:
        out = pd.Series(changepoints, name="changepoints", dtype=int)
    elif fmt == "dense" and labels == "int_label":
        out = changepoints_to_labels(changepoints, len(X_index))
        out = pd.Series(out, index=X_index, name="int_label", dtype=int)
    elif fmt == "dense" and labels == "indicator":
        out = pd.Series(False, index=X_index, name="indicator", dtype=bool)
        out.iloc[changepoints] = True
    elif labels == "score":
        # There is no sparse version of 'score'.
        # The scores are formatted in each class' _predict method, as what is a good
        # format for the scores is method dependent.
        out = scores
    return out
