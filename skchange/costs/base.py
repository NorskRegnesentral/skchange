"""Cost functions as interval evaluators."""

import numpy as np

from skchange.base import BaseIntervalScorer


class BaseCost(BaseIntervalScorer):
    """Base class template for cost functions.

    This is a common base class for cost functions. It is used to evaluate a cost
    function on a set of intervals.

    Parameters
    ----------
    param : None, optional (default=None)
        If ``None``, the cost for an optimal parameter is evaluated. If not ``None``,
        the cost is evaluated for a fixed parameter. The parameter type is specific to
        each concrete cost.
    """

    expected_interval_entries = 2

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

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Defaults to 1 parameter per variable in the data. This method should be
        overwritten in subclasses if the cost function has a different number of
        parameters per variable.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p

    def _evaluate(self, cuts: np.ndarray) -> np.ndarray:
        """Evaluate the cost on a set of intervals.

        Parameters
        ----------
        cuts : np.ndarray
            A 2D array with two columns of integer location-based intervals to evaluate.
            The subsets ``X[cuts[i, 0]:cuts[i, 1]]`` for
            ``i = 0, ..., len(cuts)`` are evaluated.

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
        """
        starts, ends = cuts[:, 0], cuts[:, 1]
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
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
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
            A 2D array of costs. One row for each interval. The number of
            columns is 1 if the cost is inherently multivariate. The number of columns
            is equal to the number of columns in the input data if the cost is
            univariate. In this case, each column represents the univariate cost for
            the corresponding input data column.
        """
        raise NotImplementedError("abstract method")
