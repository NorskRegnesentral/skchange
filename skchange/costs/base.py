"""Cost functions as interval evaluators."""

import numpy as np

from skchange.base import BaseIntervalEvaluator
from skchange.utils.validation.intervals import check_array_intervals


class BaseCost(BaseIntervalEvaluator):
    """Base class template for cost functions.

    This is a common base class for cost functions. It is used to evaluate a cost
    function on a set of intervals.

    Parameters
    ----------
    param : None, optional (default=None)
        If None, the cost for an optimal parameter is evaluated. If not None, the cost
        is evaluated for a fixed parameter. The parameter type is specific to each
        concrete cost.
    """

    def __init__(self, param=None):
        self.param = param
        super().__init__()

    def _check_param(self, param, X):
        """Check the parameter with respect to the input data.

        This method should be called in `_fit` of subclasses.
        """
        if param is None:
            return None
        return self._check_fixed_param(param, X)

    def _check_fixed_param(self, param, X):
        """Check the fixed parameter with respect to the input data.

        This method defaults to no checking, but it should be overwritten in subclasses
        to make sure `param` is valid relative to the input data `X`.
        """
        return param

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate."""
        return 1

    def _check_intervals(self, intervals: np.ndarray) -> np.ndarray:
        """Check the intervals for cost functions.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[intervals[i, 0]:intervals[i, 1]] for
            i = 0, ..., len(intervals) are evaluated.

        Returns
        -------
        intervals : np.ndarray
            The unmodified input intervals array.

        Raises
        ------
        ValueError
            If the intervals are not compatible.
        """
        return check_array_intervals(intervals, min_size=self.min_size, last_dim_size=2)

    def _evaluate(self, intervals: np.ndarray) -> np.ndarray:
        """Evaluate on a set of intervals.

        Parameters
        ----------
        intervals : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets X[intervals[i, 0]:intervals[i, 1]] for
            i = 0, ..., len(intervals) are evaluated.

        Returns
        -------
        costs : np.ndarray
            One cost value for each interval.
        """
        starts, ends = intervals[:, 0], intervals[:, 1]
        if self.param is None:
            costs = self._evaluate_optim_param(starts, ends)
        else:
            costs = self._evaluate_fixed_param(starts, ends)

        return costs

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            Costs for each interval.
        """
        raise NotImplementedError("abstract method")

    def _evaluate_fixed_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the fixed parameter.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            Costs for each interval.
        """
        raise NotImplementedError("abstract method")
