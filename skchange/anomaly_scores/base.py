"""Base classes for anomaly scores."""

import numpy as np

from skchange.base import BaseIntervalEvaluator
from skchange.utils.validation.intervals import check_array_intervals


class BaseSaving(BaseIntervalEvaluator):
    """Base class template for savings.

    A saving is a measure of the difference between a cost with a fixed baseline
    parameter and an optimised cost over an interval. Most commonly, the baseline
    parameter is pre-calculated robustly over the entire dataset under the assumption
    that anomalies are rare. Each saving thus represents the potential cost reduction if
    the parameter was optimised for the interval.
    """

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 1

    def _check_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """Check the intervals for savings.

        Parameters
        ----------
        intervals : array-like
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[intervals[i, 0]:intervals[i, 1]] for
            i = 0, ..., len(intervals) are evaluated.

        Returns
        -------
        intervals : np.ndarray
            The unmodified input intervals array.

        Raises
        ------
        ValueError
            If the intervals are not compatible.

        """
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=2)
