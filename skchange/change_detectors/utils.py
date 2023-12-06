"""Utility functions for change detection."""

import numpy as np


def changepoints_to_labels(changepoints: list):
    """Convert a list of changepoints to a list of labels.

    Parameters
    ----------
    changepoints : list
        List of changepoint indices.

    Returns
    -------
    labels : list
        List of labels: 0 for the first segment, 1 for the second, etc.
    """
    n = changepoints[-1] + 1  # Last changepoint is always the last index.
    changepoints = [-1] + changepoints  # To simplify the loop.
    labels = np.zeros(n)
    for i in range(len(changepoints) - 1):
        labels[changepoints[i] + 1 : changepoints[i + 1] + 1] = i
    return labels
