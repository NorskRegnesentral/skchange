"""Saving-type anomaly scores."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from skchange.anomaly_scores.base import BaseLocalAnomalyScore, BaseSaving
from skchange.costs import BaseCost, L2Cost
from skchange.utils.validation.data import as_2d_array


def to_saving(evaluator: Union[BaseCost, BaseSaving]) -> BaseSaving:
    """Convert a cost function to a saving.

    Parameters
    ----------
    evaluator : BaseCost or BaseSaving
        The evalutor to convert to a saving. If a cost, it must be a cost with a fixed
        parameter. If a saving is provided, it is returned as is.

    Returns
    -------
    saving : BaseSaving
        The saving based on the cost function.
    """
    if isinstance(evaluator, BaseCost):
        saving = Saving(evaluator)
    elif isinstance(evaluator, BaseSaving):
        saving = evaluator
    else:
        raise ValueError(
            f"evaluator must be an instance of BaseSaving or BaseCost. "
            f"Got {type(evaluator)}."
        )
    return saving


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

    def __init__(self, baseline_cost: BaseCost = L2Cost(param=0.0)):
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
            A 2D array of savings. One row for each interval. The number of
            columns is 1 if the saving is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the saving is
            univariate. In this case, each column represents the univariate saving for
            the corresponding input data column.
        """
        baseline_costs = self.baseline_cost.evaluate(intervals)
        optimised_costs = self.optimised_cost.evaluate(intervals)
        savings = baseline_costs - optimised_costs
        return savings

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


def to_local_anomaly_score(
    evaluator: Union[BaseCost, BaseLocalAnomalyScore],
) -> BaseLocalAnomalyScore:
    """Convert a cost function to a local anomaly score.

    Parameters
    ----------
    evaluator : BaseCost or BaseLocalAnomalyScore
        The evalutor to convert to a local anomaly score. If a cost, it must be a cost
        with a fixed parameter. If a local anomaly score is provided, it is returned as
        is.

    Returns
    -------
    local_anomaly_score : BaseLocalAnomalyScore
        The local anomaly score based on the cost function.
    """
    if isinstance(evaluator, BaseCost):
        local_anomaly_score = LocalAnomalyScore(evaluator)
    elif isinstance(evaluator, BaseLocalAnomalyScore):
        local_anomaly_score = evaluator
    else:
        raise ValueError(
            f"evaluator must be an instance of BaseLocalAnomalyScore or BaseCost. "
            f"Got {type(evaluator)}."
        )
    return local_anomaly_score


class LocalAnomalyScore(BaseLocalAnomalyScore):
    """Local anomaly scores based on costs.

    Local anomaly scores compare the data behaviour of an inner interval with the
    surrounding data contained in an outer interval. In other words, the null
    hypothesis within each outer interval is that the data is stationary, while the
    alternative hypothesis is that there is a collective anomaly within the
    outer interval.

    Parameters
    ----------
    cost : BaseCost
        The cost function to evaluate data subsets.

    Notes
    -----
    Using costs to generate local anomaly scores will be significantly slower than using
    anomaly scores that are implemented directly. This is because the local anomaly
    score requires evaluating the cost at disjoint subsets of the data
    (before and after an anomaly), which is not a natural operation for costs
    implemented as interval evaluators. It is only possible by refitting the cost
    function on the surrounding data for each interval, which is computationally
    expensive.
    """

    def __init__(self, cost: BaseCost = L2Cost()):
        self.cost = cost
        super().__init__()

        self._interval_cost = cost
        self._any_subset_cost: BaseCost = cost.clone()

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        if self.cost.min_size is None:
            return None
        else:
            return 2 * self.cost.min_size

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
        self._interval_cost.fit(X)
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
            A 2D array of savings. One row for each interval. The number of
            columns is 1 if the saving is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the saving is
            univariate. In this case, each column represents the univariate saving for
            the corresponding input data column.
        """
        X = as_2d_array(self._X)

        inner_intervals = intervals[:, 1:3]
        outer_intervals = intervals[:, [0, 3]]
        inner_costs = self._interval_cost.evaluate(inner_intervals)
        outer_costs = self._interval_cost.evaluate(outer_intervals)

        surrounding_costs = np.zeros_like(outer_costs)
        for i, interval in enumerate(intervals):
            before_inner_interval = interval[0:2]
            after_inner_interval = interval[2:4]

            before_data = X[before_inner_interval[0] : before_inner_interval[1]]
            after_data = X[after_inner_interval[0] : after_inner_interval[1]]
            surrounding_data = np.concatenate((before_data, after_data))
            self._any_subset_cost.fit(surrounding_data)
            surrounding_costs[i] = self._any_subset_cost.evaluate(
                [0, surrounding_data.shape[0]]
            )

        anomaly_scores = outer_costs - (inner_costs + surrounding_costs)
        return np.array(anomaly_scores)

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
