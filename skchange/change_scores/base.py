"""Scores for change detection."""

from skchange.base import BaseIntervalScorer
from skchange.utils.validation.interface import overrides


class BaseChangeScore(BaseIntervalScorer):
    """Base class template for change scores.

    Change scores are used to detect changes in a time series or sequence by evaluating
    the difference in distribution or data characteristics before and after a potential
    changepoint.
    """

    @property
    @overrides(BaseIntervalScorer)
    def expected_cut_entries(self) -> int:
        """Number of expected entries in the cuts array of `evaluate`."""
        return 3

    def __init__(self):
        super().__init__()
