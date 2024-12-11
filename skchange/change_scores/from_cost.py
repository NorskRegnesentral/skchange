"""Cost-based change scores."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from skchange.change_scores.base import BaseChangeScore
from skchange.costs.base import BaseCost


def to_change_score(scorer: Union[BaseCost, BaseChangeScore]) -> BaseChangeScore:
    """Convert compatible scorers to a change score.

    Parameters
    ----------
    scorer : BaseCost or BaseChangeScore
        The scorer to convert to a change score. If a change score is provided, it is
        returned as is.

    Returns
    -------
    change_score : BaseChangeScore
        The change score based on the cost function.
    """
    if isinstance(scorer, BaseCost):
        change_score = ChangeScore(scorer)
    elif isinstance(scorer, BaseChangeScore):
        change_score = scorer
    else:
        raise ValueError(
            f"scorer must be an instance of BaseChangeScore or BaseCost. "
            f"Got {type(scorer)}."
        )
    return change_score


class ChangeScore(BaseChangeScore):
    """Change score based on a cost class.

    The change score is calculated as cost difference under a no-change hypothesis
    versus a single change hypothesis.

    Parameters
    ----------
    cost : BaseCost
        The cost function.
    """

    def __init__(self, cost: BaseCost):
        self.cost = cost
        self.evaluation_type = self.cost.evaluation_type
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum valid size of an interval to evaluate."""
        return self.cost.min_size

    def _fit(self, X: ArrayLike, y=None):
        """Fit the change score.

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

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the change score for a split within an interval.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with three columns of integer locations.
            The first column is the start, the second is the split, and the third is
            the end of the interval to evaluate.
            The difference between subsets X[start:split] and X[split:end] is evaluated
            for each row in cuts.

        Returns
        -------
        scores : np.ndarray
            A 2D array of change scores. One row for each cut. The number of
            columns is 1 if the change score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the score is
            univariate. In this case, each column represents the univariate score for
            the corresponding input data column.
        """
        left_intervals = cuts[:, [0, 1]]
        right_intervals = cuts[:, [1, 2]]
        full_intervals = cuts[:, [0, 2]]
        left_costs = self.cost.evaluate(left_intervals)
        right_costs = self.cost.evaluate(right_intervals)
        no_change_costs = self.cost.evaluate(full_intervals)
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
            There are currently no reserved values for interval scorers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from skchange.costs import GaussianCost, L2Cost

        params = [
            {"cost": L2Cost()},
            {"cost": GaussianCost()},
        ]
        return params
