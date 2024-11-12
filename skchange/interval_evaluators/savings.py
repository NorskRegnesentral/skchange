"""Module for calculating savings.

Savings are cost differences between a fixed parameter representing the baseline
data behaviour and an optimised parameter over the same interval.
"""

import numpy as np
from numpy.typing import ArrayLike

from skchange.interval_evaluators.base import BaseIntervalEvaluator
from skchange.interval_evaluators.costs import BaseCost
from skchange.interval_evaluators.utils import check_array_intervals


class BaseSaving(BaseIntervalEvaluator):
    """Base class template for saving functions."""

    def __init__(self):
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 1

    def _check_intervals(self, intervals: ArrayLike) -> np.ndarray:
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=2)


class Saving(BaseSaving):
    """Saving based on a cost class.

    The saving is calculated as the difference between a baseline cost with a fixed
    parameter and the optimised cost over the same interval.

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
        self.optimised_cost = baseline_cost.clone().set_params({"param": None})
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return self.optimised_cost.min_size

    def _fit(self, X: ArrayLike, y=None):
        self.baseline_cost.fit(X)
        self.optimised_cost.fit(X)
        return self

    def _evaluate(self, intervals: np.ndarray) -> np.ndarray:
        starts, ends = intervals[:, 0], intervals[:, 1]
        baseline_costs = self.baseline_cost.evaluate(starts, ends)
        optimised_costs = self.optimised_cost.evaluate(starts, ends)
        return baseline_costs - optimised_costs
