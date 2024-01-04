"""Utility functions for anomaly detection."""

from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def merge_anomalies(
    collective_anomalies: Union[
        List[Tuple[int, int]], List[Tuple[int, int, np.ndarray]]
    ] = None,
    point_anomalies: Union[
        List[int],
        List[Tuple[int, int]],
        List[Tuple[int, np.ndarray]],
        List[Tuple[int, int, np.ndarray]],
    ] = None,
) -> List[Tuple[int, int, np.ndarray]]:
    """Merge collective and point anomalies into a single list of intervals.

    Parameters
    ----------
    collective_anomalies : list, optional (default=None)
        List of tuples containing inclusive start and end indices of collective
        anomalies.
    point_anomalies : list, optional (default=None)
        List of point anomaly indices.

    Returns
    -------
    list
        List of tuples containing inclusive start and end indices of collective
        anomalies and point anomalies.
    """
    if collective_anomalies is None and point_anomalies is None:
        raise ValueError(
            "Either collective_anomalies or point_anomalies must be given."
        )

    anomalies = []
    if collective_anomalies:
        anomalies += collective_anomalies
    if point_anomalies:
        # Convert point anomalies to the same format as collective anomalies
        if isinstance(point_anomalies[0], int):
            anomalies += [(i, i) for i in point_anomalies]
        elif len(point_anomalies[0]) == 2 and isinstance(
            point_anomalies[0][-1], np.ndarray
        ):
            anomalies += [(i, i, components) for (i, components) in point_anomalies]
        else:
            anomalies += point_anomalies

    anomalies = sorted(anomalies)
    return anomalies


def anomalies_to_labels(
    anomalies: List[Tuple[int, int]], n: int, p: int = None
) -> np.ndarray:
    """Convert anomaly indices to labels.

    Parameters
    ----------
    anomalies : list
        List of tuples containing inclusive start and end indices of  collective
        anomalies and point anomalies.
    n : int
        Sample size.
    p : int
        Dimensionality of the data input to the anomaly detector.

    Returns
    -------
    np.ndarray
        Array of labels, where 0 is the normal class, and 1, 2, ... are labels for each
        distinct collective and/or point_anomaly.
    """
    labels = np.zeros(n, dtype=int) if p is None else np.zeros((n, p), dtype=int)
    if len(anomalies) == 0:
        return labels

    if len(anomalies[0]) == 2:
        for i, (start, end) in enumerate(anomalies):
            labels[start : end + 1] = i + 1
    elif len(anomalies[0]) == 3:
        # Multivariate
        for i, (start, end, components) in enumerate(anomalies):
            labels[start : end + 1, components] = i + 1
    return labels


def format_anomaly_output(
    fmt: str,
    labels: str,
    X_index: pd.Index,
    collective_anomalies: List[tuple] = None,
    point_anomalies: List[tuple] = None,
    scores: Union[pd.Series, pd.DataFrame] = None,
) -> pd.Series:
    """Format the predict method output of change detectors.

    Parameters
    ----------
    fmt : str
        Format of the output. Either "sparse" or "dense".
    labels : str
        Labels of the output. Either "indicator", "score" or "int_label".
    X_index : pd.Index
        Index of the input data.
    collective_anomalies : list, optional (default=None)
        List of tuples containing inclusive start and end indices of collective
        anomalies.
    point_anomalies : list, optional (default=None)
        List of point anomaly indices.
    scores : pd.Series or pd.DataFrame, optional (default=None)
        Series or DataFrame of scores. If Series, it must be named 'score', and if
        DataFrame, it must have a column named 'score'.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
    n = X_index.size
    anomalies = merge_anomalies(collective_anomalies, point_anomalies)
    if labels == "int_label":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n)
            out = pd.Series(anomaly_labels, index=X_index, name="int_label", dtype=int)
        elif fmt == "sparse":
            out = pd.DataFrame(anomalies, columns=["start", "end"])
    elif labels == "indicator":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n)
            out = pd.Series(anomaly_labels > 0, index=X_index, name="indicator")
        elif fmt == "sparse":
            out = pd.DataFrame(anomalies, columns=["start", "end"])
    elif labels == "score":
        # There is no sparse version of 'score'.
        # The scores are formatted in each class' _predict method, as what is a good
        # format for the scores is method dependent.
        out = scores
    return out


def format_multivariate_anomaly_output(
    fmt: str,
    labels: str,
    X_index: pd.Index,
    X_columns: pd.Index,
    collective_anomalies: List[dict] = None,
    point_anomalies: List[dict] = None,
    scores: Union[pd.Series, pd.DataFrame] = None,
) -> pd.Series:
    """Format the predict method output of change detectors.

    Parameters
    ----------
    fmt : str
        Format of the output. Either "sparse" or "dense".
    labels : str
        Labels of the output. Either "indicator", "score" or "int_label".
    X_index : pd.Index
        Index of the input data.
    X_columns : pd.Index
        Columns of the input data.
    collective_anomalies : list, optional (default=None)
        List of tuples containing inclusive start and end indices of collective
        anomalies.
    point_anomalies : list, optional (default=None)
        List of point anomaly indices.
    scores : pd.Series or pd.DataFrame, optional (default=None)
        Series or DataFrame of scores. If Series, it must be named 'score', and if
        DataFrame, it must have a column named 'score'.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
    n = X_index.size
    p = X_columns.size
    anomalies = merge_anomalies(collective_anomalies, point_anomalies)
    if labels == "int_label":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n, p)
            out = pd.DataFrame(
                anomaly_labels, index=X_index, columns=X_columns, dtype=int
            )
        elif fmt == "sparse":
            out = pd.DataFrame(anomalies, columns=["start", "end", "components"])
    elif labels == "indicator":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n, p)
            out = pd.DataFrame(anomaly_labels > 0, index=X_index, columns=X_columns)
        elif fmt == "sparse":
            out = pd.DataFrame(anomalies, columns=["start", "end", "components"])
    elif labels == "score":
        # There is no sparse version of 'score'.
        # The scores are formatted in each class' _predict method, as what is a good
        # format for the scores is method dependent.
        out = scores
    return out
