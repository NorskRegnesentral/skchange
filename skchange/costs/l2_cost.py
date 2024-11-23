"""L2 cost."""

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from skchange.costs.base import BaseCost
from skchange.costs.utils import MeanType, check_mean
from skchange.utils.numba import njit
from skchange.utils.numba.stats import col_cumsum
from skchange.utils.validation.data import as_2d_array


@njit
def l2_cost_optim(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.
    sums : `np.ndarray`
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : `np.ndarray`
        Cumulative sum of the squared input data, with a row of 0-entries as the first
        row.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of
        columns is 1 if the cost is inherently multivariate. The number of columns
        is equal to the number of columns in the input data if the cost is
        univariate. In this case, each column represents the univariate cost for
        the corresponding input data column.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - partial_sums**2 / n
    return costs


@njit
def l2_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    sums: np.ndarray,
    sums2: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for a fixed constant mean for each segment.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.
    sums : `np.ndarray`
        Cumulative sum of the input data, with a row of 0-entries as the first row.
    sums2 : `np.ndarray`
        Cumulative sum of the squared input data, with a row of 0-entries as the first
        row.
    mean : `np.ndarray`
        Fixed mean for the cost calculation.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of
        columns is 1 if the cost is inherently multivariate. The number of columns
        is equal to the number of columns in the input data if the cost is
        univariate. In this case, each column represents the univariate cost for
        the corresponding input data column.
    """
    partial_sums = sums[ends] - sums[starts]
    partial_sums2 = sums2[ends] - sums2[starts]
    n = (ends - starts).reshape(-1, 1)
    costs = partial_sums2 - 2 * mean * partial_sums + n * mean**2
    return costs


class L2Cost(BaseCost):
    """L2 cost of a constant mean.

    Parameters
    ----------
    param : float or array-like, optional (default=None)
        Fixed mean for the cost calculation. If None, the optimal mean is calculated.
    """

    def __init__(self, param: Union[MeanType, None] = None):
        super().__init__(param)

    def _check_fixed_param(self, param: MeanType, X: np.ndarray) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : float or array-like
            The input parameter to check.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        return check_mean(param, X)

    def _fit(self, X: ArrayLike, y=None):
        """Fit the cost interval evaluator.

        This method precomputes quantities that speed up the cost evaluation.

        Parameters
        ----------
        X : array-like
            Input data.
        y: None
            Ignored. Included for API consistency by convention.
        """
        X = as_2d_array(X)
        self._mean = self._check_param(self.param, X)

        self.sums_ = col_cumsum(X, init_zero=True)
        self.sums2_ = col_cumsum(X**2, init_zero=True)

        return self

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
        return l2_cost_optim(starts, ends, self.sums_, self.sums2_)

    def _evaluate_fixed_param(self, starts, ends):
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
        return l2_cost_fixed(starts, ends, self.sums_, self.sums2_, self._mean)

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
        params = [
            {"param": None},
            {"param": 0.0},
        ]
        return params
