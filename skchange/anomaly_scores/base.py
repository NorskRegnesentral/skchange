"""Base classes for anomaly scores."""

import numpy as np
from numpy.typing import ArrayLike

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

    def _check_intervals(self, intervals: ArrayLike) -> np.ndarray:
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=2)
