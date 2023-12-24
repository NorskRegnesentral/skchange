"""Utility functions for anomaly detection."""

from typing import List, Tuple

import numpy as np
import pandas as pd


def merge_anomalies(
    collective_anomalies: List[Tuple[int, int, np.ndarray]] = None,
    point_anomalies: List[Tuple[int, np.ndarray]] = None,
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
        if isinstance(point_anomalies[0], tuple):
            # Multivariate
            anomalies += [(i, i, components) for (i, components) in point_anomalies]
        else:
            # Univariate
            anomalies += [(i, i) for i in point_anomalies]
    anomalies = sorted(anomalies)
    return anomalies


def anomalies_to_labels(
    anomalies: List[Tuple[int, int]], n: int, p: int = None
) -> np.ndarray:
    """Convert anomaly indices to labels.

    Parameters
    ----------
    n : int
        Sample size.
    p : int
        Dimensionality of the data input to the anomaly detector.
    anomalies : list
        List of tuples containing inclusive start and end indices of collective
        anomalies and point anomalies.

    Returns
    -------
    np.ndarray
        Array of labels, where 0 is the normal class, and 1, 2, ... are labels for each
        distinct collective and/or point_anomaly.
    """
    if len(anomalies[0]) == 2:
        labels = np.zeros(n, dtype=int)
        for i, (start, end) in enumerate(anomalies):
            labels[start : end + 1] = i + 1
    elif len(anomalies[0]) == 3:
        # Multivariate
        labels = np.zeros((n, p), dtype=int)
        for i, (start, end, components) in enumerate(anomalies):
            labels[start : end + 1, components] = i + 1
    return labels


def format_anomaly_output(
    fmt: str,
    labels: str,
    n: int,
    collective_anomalies: List[dict] = None,
    point_anomalies: List[dict] = None,
    X_index: pd.Index = None,
    scores: np.ndarray = None,
) -> pd.Series:
    """Format the predict method output of change detectors.

    Parameters
    ----------
    fmt : str
        Format of the output. Either "sparse" or "dense".
    labels : str
        Labels of the output. Either "indicator", "score" or "int_label".
    n : int
        Sample size of the data input to the anomaly detector.
    collective_anomalies : list, optional (default=None)
        List of tuples containing inclusive start and end indices of collective
        anomalies.
    point_anomalies : list, optional (default=None)
        List of point anomaly indices.
    X_index : pd.Index
        Index of the input data.
    scores : np.ndarray, optional (default=None)
        Array of scores.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
    anomalies = merge_anomalies(collective_anomalies, point_anomalies)
    if labels == "int_label":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n)
            out = pd.Series(anomaly_labels, index=X_index, name="int_label", dtype=int)
        elif fmt == "sparse":
            out = pd.Series([pd.Interval(*anom, closed="both") for anom in anomalies])
    elif labels == "indicator":
        if fmt == "dense":
            anomaly_labels = anomalies_to_labels(anomalies, n)
            out = pd.Series(anomaly_labels > 0, index=X_index, name="indicator")
        elif fmt == "sparse":
            out = pd.Series([pd.Interval(*anom, closed="both") for anom in anomalies])
    elif labels == "score":
        if fmt == "dense":
            out = pd.Series(scores, index=X_index, name="score", dtype=float)
        elif fmt == "sparse":
            out = pd.Series(
                index=pd.IntervalIndex.from_tuples(anomalies, closed="both"),
                data=scores[[end for _, end in anomalies]],
            )
    return out


def format_multivariate_anomaly_output(
    fmt: str,
    labels: str,
    n: int,
    p: int,
    collective_anomalies: List[dict] = None,
    point_anomalies: List[dict] = None,
    X_index: pd.Index = None,
    X_columns: pd.Index = None,
    scores: np.ndarray = None,
) -> pd.Series:
    """Format the predict method output of change detectors.

    Parameters
    ----------
    fmt : str
        Format of the output. Either "sparse" or "dense".
    labels : str
        Labels of the output. Either "indicator", "score" or "int_label".
    n : int
        Sample size of the data input to the anomaly detector.
    p : int
        Dimensionality of the data input to the anomaly detector.
    collective_anomalies : list, optional (default=None)
        List of tuples containing inclusive start and end indices of collective
        anomalies.
    point_anomalies : list, optional (default=None)
        List of point anomaly indices.
    X_index : pd.Index
        Index of the input data.
    scores : np.ndarray, optional (default=None)
        Array of scores.

    Returns
    -------
    pd.Series
        Either a sparse or dense pd.Series of boolean labels, integer labels or scores.
    """
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
            out = pd.DataFrame(
                anomaly_labels > 0, index=X_index, columns=X_columns, dtype=int
            )
        elif fmt == "sparse":
            out = pd.DataFrame(anomalies, columns=["start", "end", "components"])
    elif labels == "score":
        if fmt == "dense":
            out = pd.Series(scores, index=X_index, name="score", dtype=float)
        elif fmt == "sparse":
            out = pd.Series(
                index=pd.IntervalIndex.from_tuples(anomalies, closed="both"),
                data=scores[[end for _, end in anomalies]],
            )
    return out
