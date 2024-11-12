"""Utility functions for interval evaluators."""

import numpy as np
from numpy.typing import ArrayLike

from skchange.utils.validation.data import as_2d_array


def check_array_intervals(
    intervals: ArrayLike,
    min_size: int,
    last_dim_size: int,
) -> np.ndarray:
    """Check array type intervals."""
    intervals = as_2d_array(intervals, vector_as_column=False)

    if not np.issubdtype(intervals.dtype, np.integer):
        raise ValueError("The intervals must be of integer type.")

    if intervals.shape[-1] != last_dim_size:
        raise ValueError(
            (
                "The intervals must be specified as an array with length",
                f" {last_dim_size} in the last dimension.",
            )
        )

    if not np.all(np.diff(intervals, axis=1) > 0):
        raise ValueError(
            (
                "All rows in the intervals must be strictly increasing.",
                "The first and last columns represent the starts and ends of the",
                "intervals respectively.",
            )
        )

    interval_sizes = intervals[:, -1] - intervals[:, 0]
    if np.any(interval_sizes < min_size):
        raise ValueError(
            (
                f"The interval sizes must be at least {min_size}."
                f" Found an interval with size {np.min(interval_sizes)}.",
            )
        )

    return intervals
