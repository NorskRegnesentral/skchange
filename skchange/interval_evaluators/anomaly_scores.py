"""Module for calculating anomaly scores."""

import numpy as np
from numpy.typing import ArrayLike

from skchange.interval_evaluators.base import BaseIntervalEvaluator
from skchange.interval_evaluators.costs import BaseCost
from skchange.interval_evaluators.utils import check_array_intervals


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


class Saving(BaseSaving):
    """Saving based on a cost class.

    Savings represent the difference between a cost based on a fixed baseline parameter
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
