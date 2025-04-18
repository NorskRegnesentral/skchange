"""Scores for change detection."""

from ..base import BaseIntervalScorer


class BaseChangeScore(BaseIntervalScorer):
    """Base class template for change scores.

    Change scores are used to detect changes in a time series or sequence by evaluating
    the difference in distribution or data characteristics before and after a potential
    changepoint.
    """

    expected_cut_entries = 3

    def __init__(self):
        super().__init__()
