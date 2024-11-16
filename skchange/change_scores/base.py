"""Scores for change detection."""

from skchange.base import BaseIntervalEvaluator


class BaseChangeScore(BaseIntervalEvaluator):
    """Base class template for change scores.

    Change scores are used to detect changes in a time series or sequence by evaluating
    the difference in distribution or data characteristics before and after a potential
    changepoint.
    """

    expected_interval_entries = 3

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 2
