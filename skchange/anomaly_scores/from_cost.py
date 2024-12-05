"""Saving-type anomaly scores."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from skchange.anomaly_scores.base import BaseLocalAnomalyScore, BaseSaving
from skchange.costs import BaseCost, L2Cost
from skchange.utils.validation.cuts import check_cuts_array
from skchange.utils.validation.data import as_2d_array


def to_saving(scorer: Union[BaseCost, BaseSaving]) -> BaseSaving:
    """Convert compatible scorers to a saving.

    Parameters
    ----------
    scorer : BaseCost or BaseSaving
        The scorer to convert to a saving. If a cost, it must be a cost with a fixed
        parameter. If a saving is provided, it is returned as is.

    Returns
    -------
    saving : BaseSaving
        The saving based on the cost function.
    """
    if isinstance(scorer, BaseCost):
        saving = Saving(scorer)
    elif isinstance(scorer, BaseSaving):
        saving = scorer
    else:
        raise ValueError(
            f"scorer must be an instance of BaseSaving or BaseCost. "
            f"Got {type(scorer)}."
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
        The baseline cost with a fixed parameter. The optimised cost is
        constructed by copying the baseline cost and setting the parameter to None.
    """

    def __init__(self, baseline_cost: BaseCost):
        if baseline_cost.param is None:
            raise ValueError("The baseline cost must have a fixed parameter.")

        self.baseline_cost = baseline_cost
        self.optimised_cost: BaseCost = baseline_cost.clone().set_params(param=None)
        self.evaluation_type = self.baseline_cost.evaluation_type
        super().__init__()

    @property
    def min_size(self) -> int:
        """Minimum valid size of the interval to evaluate."""
        return self.optimised_cost.min_size

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the saving function.

        Defaults to 1 parameter per variable in the data. This method should be
        overwritten in subclasses if the cost function has a different number of
        parameters per variable.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return self.optimised_cost.get_param_size(p)

    def _fit(self, X: ArrayLike, y=None):
        """Fit the saving scorer.

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

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the saving on a set of intervals.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[cuts[i, 0]:cuts[i, 1]] for
            i = 0, ..., len(cuts) are evaluated.

        Returns
        -------
        savings : np.ndarray
            A 2D array of savings. One row for each interval. The number of
            columns is 1 if the saving is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the saving is
            univariate. In this case, each column represents the univariate saving for
            the corresponding input data column.
        """
        baseline_costs = self.baseline_cost.evaluate(cuts)
        optimised_costs = self.optimised_cost.evaluate(cuts)
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
            There are currently no reserved values for interval scorers.

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
    scorer: Union[BaseCost, BaseLocalAnomalyScore],
) -> BaseLocalAnomalyScore:
    """Convert compatible scorers to a saving.

    Parameters
    ----------
    scorer : BaseCost or BaseLocalAnomalyScore
        The scorer to convert to a local anomaly score. If a local anomaly score is
        provided, it is returned as is.

    Returns
    -------
    local_anomaly_score : BaseLocalAnomalyScore
        The local anomaly score based on the cost function.
    """
    if isinstance(scorer, BaseCost):
        local_anomaly_score = LocalAnomalyScore(scorer)
    elif isinstance(scorer, BaseLocalAnomalyScore):
        local_anomaly_score = scorer
    else:
        raise ValueError(
            f"scorer must be an instance of BaseLocalAnomalyScore or BaseCost. "
            f"Got {type(scorer)}."
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
    implemented as interval scorers. It is only possible by refitting the cost
    function on the surrounding data for each cut, which is computationally
    expensive.
    """

    def __init__(self, cost: BaseCost):
        self.cost = cost
        self.evaluation_type = self.cost.evaluation_type
        super().__init__()

        self._interval_cost = cost
        self._any_subset_cost: BaseCost = cost.clone()

    @property
    def min_size(self) -> int:
        """Minimum valid size of the interval to evaluate."""
        return self.cost.min_size

    def _fit(self, X: ArrayLike, y=None):
        """Fit the saving scorer.

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

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the local anomaly score on sets of inner and outer intervals.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with two columns of integer location-based cuts to evaluate.
            The first column is the start of the outer interval, the second column is
            the start of the inner interval, the third column is the end of the inner
            interval, and the fourth column is the end of the outer interval.

        Returns
        -------
        savings : np.ndarray
            A 2D array of anomaly scores. One row for each interval. The number of
            columns is 1 if the anomaly score is inherently multivariate. The number of
            columns is equal to the number of columns in the input data if the anomaly
            score is univariate. In this case, each column represents the univariate
            anomaly score for the corresponding input data column.
        """
        X = as_2d_array(self._X)

        inner_intervals = cuts[:, 1:3]
        outer_intervals = cuts[:, [0, 3]]
        inner_costs = self._interval_cost.evaluate(inner_intervals)
        outer_costs = self._interval_cost.evaluate(outer_intervals)

        surrounding_costs = np.zeros_like(outer_costs)
        for i, interval in enumerate(cuts):
            before_inner_interval = interval[0:2]
            after_inner_interval = interval[2:4]

            before_data = X[before_inner_interval[0] : before_inner_interval[1]]
            after_data = X[after_inner_interval[0] : after_inner_interval[1]]
            surrounding_data = np.concatenate((before_data, after_data))
            self._any_subset_cost.fit(surrounding_data)
            surrounding_costs[i] = self._any_subset_cost.evaluate(
                np.array([0, surrounding_data.shape[0]])
            )

        anomaly_scores = outer_costs - (inner_costs + surrounding_costs)
        return np.array(anomaly_scores)

    def _check_cuts(self, cuts: np.ndarray) -> np.ndarray:
        """Check cuts for compatibility.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array of integer location-based cuts to evaluate. Each row in the array
            must be sorted in increasing order.

        Returns
        -------
        cuts : np.ndarray
            The unmodified input cuts array.

        Raises
        ------
        ValueError
            If the cuts are not compatible.
        """
        cuts = check_cuts_array(
            cuts,
            last_dim_size=self.expected_cut_entries,
        )

        inner_intervals_sizes = np.diff(cuts[:, [1, 2]], axis=1)
        before_inner_sizes = np.diff(cuts[:, [0, 1]], axis=1)
        after_inner_sizes = np.diff(cuts[:, [2, 3]], axis=1)
        surrounding_intervals_sizes = before_inner_sizes + after_inner_sizes
        if not np.all(inner_intervals_sizes >= self.min_size):
            raise ValueError(
                f"The inner intervals must be at least min_size={self.min_size}."
                f" Got {inner_intervals_sizes.min()}."
            )
        if not np.all(surrounding_intervals_sizes >= self.min_size):
            raise ValueError(
                f"The surrounding intervals must be at least min_size={self.min_size}."
                f" in total. Got {surrounding_intervals_sizes.min()}."
            )

        return cuts

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
        from skchange.costs import GaussianVarCost

        params = [
            {"cost": L2Cost()},
            {"cost": GaussianVarCost()},
        ]
        return params
