"""Saving-type anomaly scores."""

import numpy as np
from numpy.typing import ArrayLike

from skchange.anomaly_scores.base import BaseSaving
from skchange.costs.base import BaseCost


class Saving(BaseSaving):
    """Saving based on a cost class.

    Savings are the difference between a cost based on a fixed baseline parameter
    and an optimized cost over a given interval. The baseline parameter must be robustly
    estimated across the entire dataset, assuming that anomalies are rare. Each saving
    indicates the potential for cost reduction if the parameter were optimized for that
    specific interval.

    Parameters
    ----------
    baseline_cost : BaseCost
        The baseline cost function with a fixed parameter. The optimised cost is
        constructed by copying the baseline cost and setting the parameter to None.
    """

    def __init__(self, baseline_cost: BaseCost):
        if baseline_cost.param is None:
            raise ValueError("The baseline cost must have a fixed parameter.")

        self.baseline_cost = baseline_cost
        self.optimised_cost = baseline_cost.clone().set_params(param=None)
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return self.optimised_cost.min_size

    def _fit(self, X: ArrayLike, y=None):
        """Fit the saving evaluator.

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
        self.baseline_cost.fit(X)
        self.optimised_cost.fit(X)
        return self

    def _evaluate(self, intervals: np.ndarray) -> np.ndarray:
        """Evaluate the saving on a set of intervals.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[intervals[i, 0]:intervals[i, 1]] for
            i = 0, ..., len(intervals) are evaluated.

        Returns
        -------
        savings : np.ndarray
            Savings for each interval.
        """
        starts, ends = intervals[:, 0], intervals[:, 1]
        baseline_costs = self.baseline_cost.evaluate(starts, ends)
        optimised_costs = self.optimised_cost.evaluate(starts, ends)
        return baseline_costs - optimised_costs

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
            {"baseline_cost": L2Cost(param=0.0)},
            {"baseline_cost": L2Cost(param=np.array([1.0]))},
        ]
        return params
