"""Cost-based change scores."""

from typing import Union

from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.costs.base import BaseCost


def to_change_score(evaluator: Union[BaseCost, BaseChangeScore]) -> BaseChangeScore:
    """Convert a cost function to a change score.

    Parameters
    ----------
    evaluator : BaseCost or BaseChangeScore
        The evalutor to convert to a change score. If a change score is provided, it is
        returned as is.

    Returns
    -------
    change_score : BaseChangeScore
        The change score based on the cost function.
    """
    if isinstance(evaluator, BaseCost):
        change_score = ChangeScore(evaluator)
    elif isinstance(evaluator, BaseChangeScore):
        change_score = evaluator
    else:
        raise ValueError(
            f"evaluator must be an instance of BaseChangeScore or BaseCost. "
            f"Got {type(evaluator)}."
        )
    return change_score


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
            A 2D array of change scores. One row for each interval. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        left_costs = self.cost.evaluate(intervals[:, [0, 1]])
        right_costs = self.cost.evaluate(intervals[:, [1, 2]])
        no_change_costs = self.cost.evaluate(intervals[:, [0, 2]])
        change_scores = no_change_costs - (left_costs + right_costs)
        return change_scores

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for interval evaluators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skchange.costs.l2_cost import L2Cost

        params = [
            {"cost": L2Cost()},
            {"cost": L2Cost()},
        ]
        return params
