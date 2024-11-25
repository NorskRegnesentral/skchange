"""Utility functions for interval evaluators."""

import numpy as np


def check_array_intervals(
    intervals: np.ndarray,
    min_size: int = 1,
    last_dim_size: int = 2,
) -> np.ndarray:
    """Check array type intervals.

    Parameters
    ----------
    intervals : np.ndarray
        Array of intervals to check.
    min_size : int, optional (default=1)
        Minimum size of the intervals.
    last_dim_size : int, optional (default=2)
        Size of the last dimension.

    Returns
    -------
    intervals : np.ndarray
        The unmodified input intervals array.

    Raises
    ------
    ValueError
        If the intervals does not meet the requirements.
    """
    if intervals.ndim != 2:
        raise ValueError("The intervals must be a 2D array.")

    if not np.issubdtype(intervals.dtype, np.integer):
        raise ValueError("The intervals must be of integer type.")

    if intervals.shape[-1] != last_dim_size:
        raise ValueError(
            "The intervals must be specified as an array with length "
            f"{last_dim_size} in the last dimension."
        )

    interval_diffs = np.diff(intervals, axis=1)
    if not np.all(interval_diffs >= min_size):
        raise ValueError(
            "All rows in `intervals` must be strictly increasing and each entry must"
            f" be more than min_size={min_size} apart."
        )
    return intervals
