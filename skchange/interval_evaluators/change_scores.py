"""Scores for change detection."""

import numpy as np
from numpy.typing import ArrayLike

from skchange.interval_evaluators.base import BaseIntervalEvaluator
from skchange.interval_evaluators.costs import BaseCost
from skchange.interval_evaluators.utils import check_array_intervals


class BaseChangeScore(BaseIntervalEvaluator):
    """Base class template for change scores."""

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 2

    def _check_intervals(self, intervals: ArrayLike) -> np.ndarray:
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=3)


class ChangeScore(BaseChangeScore):
    """Change score based a cost class."""

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    def _fit(self, X: ArrayLike, y=None):
        self.cost.fit(X)
        return self

    def _evaluate(self, intervals):
        starts, splits, ends = intervals[:, 0], intervals[:, 1], intervals[:, 2]
        left_costs = self.cost.evaluate(starts, splits)
        right_costs = self.cost.evaluate(splits, ends)
        no_change_costs = self.cost.evaluate(starts, ends)
        return no_change_costs - (left_costs + right_costs)
