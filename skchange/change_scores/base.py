"""Scores for change detection."""

import numpy as np

from skchange.base import BaseIntervalEvaluator
from skchange.utils.validation.intervals import check_array_intervals


class BaseChangeScore(BaseIntervalEvaluator):
    """Base class template for change scores.

    Change scores are used to detect changes in a time series or sequence by evaluating
    the difference in distribution or data characteristics before and after a potential
    changepoint.
    """

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 2

    def _check_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """Check the intervals for change scores.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array with three columns of integer location-based intervals to
            evaluate. The difference between subsets X[intervals[i, 0]:intervals[i, 1]]
            and X[intervals[i, 1]:intervals[i, 2]] are evaluated for
            i = 0, ..., len(intervals).

        Returns
        -------
        intervals : np.ndarray
            The unmodified input intervals array.

        Raises
        ------
        ValueError
            If the intervals are not compatible.
        """
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=3)
