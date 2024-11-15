"""Cost-based change scores."""

from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.costs.base import BaseCost


class ChangeScore(BaseChangeScore):
    """Change score based on a cost class.

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
        """Fit the change score evaluator.

        Parameters
        ----------
        X : array-like
            Input data.
        y : None
            Ignored. Included for API consistency by convention.

        Returns
        -------
        self :
            Reference to self.
        """
        self.cost.fit(X)
        return self

    def _evaluate(self, intervals):
        """Evaluate the change score on a set of intervals.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array with three columns of integer location-based intervals to
            evaluate. The difference between subsets X[intervals[i, 0]:intervals[i, 1]]
            and X[intervals[i, 1]:intervals[i, 2]] are evaluated for
            i = 0, ..., len(intervals).

        Returns
        -------
        scores : np.ndarray
            Change scores for each interval.
        """
        starts, splits, ends = intervals[:, 0], intervals[:, 1], intervals[:, 2]
        left_costs = self.cost.evaluate(starts, splits)
        right_costs = self.cost.evaluate(splits, ends)
        no_change_costs = self.cost.evaluate(starts, ends)
        return no_change_costs - (left_costs + right_costs)
