"""Cost-based change scores."""

from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.interval_evaluators.costs import BaseCost


class ChangeScore(BaseChangeScore):
    """Change score based a cost class.

    The change score is calculated as the difference between the cost of an interval
    and the sum of the costs on either side of a split point within the interval.

    Parameters
    ----------
    cost : BaseCost
        The cost function to evaluate on the intervals.
    """

    def __init__(self, cost: BaseCost):
        self.cost = cost
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 2 * self.cost.min_size

    def _fit(self, X: ArrayLike, y=None):
        self.cost.fit(X)
        return self

    def _evaluate(self, intervals):
        starts, splits, ends = intervals[:, 0], intervals[:, 1], intervals[:, 2]
        left_costs = self.cost.evaluate(starts, splits)
        right_costs = self.cost.evaluate(splits, ends)
        no_change_costs = self.cost.evaluate(starts, ends)
        return no_change_costs - (left_costs + right_costs)
