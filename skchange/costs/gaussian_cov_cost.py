"""Multivariate Gaussian likelihood cost."""

__author__ = ["johannvk", "Tveten"]

from typing import Union

import numpy as np
from numpy.typing import ArrayLike

from skchange.costs.base import BaseCost
from skchange.costs.utils import CovType, MeanType, check_cov, check_mean
from skchange.utils.numba import njit, prange
from skchange.utils.numba.stats import log_det_covariance
from skchange.utils.validation.data import as_2d_array


@njit
def _gaussian_ll_at_mle_for_segment(
    X: np.ndarray,
    start: int,
    end: int,
) -> float:
    """Calculate the Gaussian log likelihood at the MLE for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).

    Returns
    -------
    mv_ll_at_mle : float
        Log likelihood of the interval
        [start, end) in the data matrix X,
        evaluated at the maximum likelihood parameter
        estimates for the mean and covariance matrix.
    """
    n = end - start
    p = X.shape[1]

    X_segment = X[start:end]
    log_det_cov = log_det_covariance(X_segment)

    if np.isnan(log_det_cov):
        raise RuntimeError(
            f"The covariance matrix of `X[{start}:{end}]` is not positive definite."
            + " Quick and dirty fix: Add a tiny amount of random noise to the data."
        )

    log_likelihood = -n * p * np.log(2 * np.pi) - n * log_det_cov - p * n
    return log_likelihood


@njit
def gaussian_cov_cost_optim(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Calculate the L2 cost for an optimal constant mean for each segment.

    Parameters
    ----------
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of
        columns is 1 if the cost is inherently multivariate. The number of columns
        is equal to the number of columns in the input data if the cost is
        univariate. In this case, each column represents the univariate cost for
        the corresponding input data column.
    """
    num_starts = len(starts)
    costs = np.zeros(num_starts).reshape(-1, 1)
    for i in prange(num_starts):
        segment_log_likelihood = _gaussian_ll_at_mle_for_segment(X, starts[i], ends[i])
        costs[i, 0] = -segment_log_likelihood
    return costs


@njit
def _gaussian_ll_at_fixed_for_segment(
    X: np.ndarray,
    start: int,
    end: int,
    mean: np.ndarray,
    log_det_cov: float,
    inv_cov: np.ndarray,
) -> float:
    """Calculate the Gaussian log likelihood at a fixed parameter for a segment.

    Parameters
    ----------
    X : np.ndarray
        Data matrix. Rows are observations and columns are variables.
    start : int
        Start index of the segment (inclusive).
    end : int
        End index of the segment (exclusive).

    Returns
    -------
    mv_ll_at_mle : float
        Log likelihood of the interval
        [start, end) in the data matrix X,
        evaluated at the maximum likelihood parameter
        estimates for the mean and covariance matrix.
    """
    n = end - start
    p = X.shape[1]

    X_segment = X[start:end]
    X_centered = X_segment - mean
    quadratic_form = np.sum(X_centered @ inv_cov * X_centered, axis=1)
    log_likelihood = (
        -n * p * np.log(2 * np.pi) - n * log_det_cov - np.sum(quadratic_form)
    )
    return log_likelihood


@njit
def gaussian_cov_cost_fixed(
    starts: np.ndarray,
    ends: np.ndarray,
    X: np.ndarray,
    mean: np.ndarray,
    log_det_cov: float,
    inv_cov: np.ndarray,
) -> np.ndarray:
    """Calculate the L2 cost for a fixed constant mean for each segment.

    Parameters
    ----------
    mean : `np.ndarray`
        Fixed mean for the cost calculation.
    starts : `np.ndarray`
        Start indices of the segments.
    ends : `np.ndarray`
        End indices of the segments.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of
        columns is 1 if the cost is inherently multivariate. The number of columns
        is equal to the number of columns in the input data if the cost is
        univariate. In this case, each column represents the univariate cost for
        the corresponding input data column.
    """
    num_starts = len(starts)
    costs = np.zeros(num_starts).reshape(-1, 1)
    for i in prange(num_starts):
        segment_log_likelihood = _gaussian_ll_at_fixed_for_segment(
            X, starts[i], ends[i], mean, log_det_cov, inv_cov
        )
        costs[i, 0] = -segment_log_likelihood
    return costs


class GaussianCovCost(BaseCost):
    """Multivariate Gaussian likelihood cost.

    Parameters
    ----------
    param : 2-tuple of float or np.ndarray, or None (default=None)
        Fixed mean and covariance matrix for the cost calculation.
        If None, the maximum likelihood estimates are used.

    """

    def __init__(self, param: Union[tuple[MeanType, CovType], None] = None):
        super().__init__(param)

    def _check_fixed_param(
        self, param: tuple[MeanType, CovType], X: np.ndarray
    ) -> np.ndarray:
        """Check if the fixed mean parameter is valid.

        Parameters
        ----------
        param : 2-tuple of float or np.ndarray
            Fixed mean and covariance matrix for the cost calculation.
        X : np.ndarray
            Input data.

        Returns
        -------
        mean : np.ndarray
            Fixed mean for the cost calculation.
        """
        mean, cov = param
        mean = check_mean(mean, X)
        cov = check_cov(cov, X)
        return mean, cov

    @property
    def min_size(self) -> Union[int, None]:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as intervals[i, -1] - intervals[i, 0].
        """
        if self.is_fitted:
            return self.X_.shape[1] + 1
        else:
            return None

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.
        """
        return p + p * (p + 1) // 2

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
        self._param = self._check_param(self.param, X)

        if self.param is not None:
            self._mean, cov = self._param
            self._inv_cov = np.linalg.inv(cov)
            _, self._log_det_cov = np.linalg.slogdet(cov)

        # Stored as np.ndarray for use in _evaluate. self._X can many types.
        self.X_ = X
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
            columns is 1 since the GaussianCovCost is inherently multivariate.
        """
        return gaussian_cov_cost_optim(starts, ends, self.X_)

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
            columns is 1 since the GaussianCovCost is inherently multivariate.
        """
        return gaussian_cov_cost_fixed(
            starts, ends, self.X_, self._mean, self._log_det_cov, self._inv_cov
        )

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
            {"param": (0.0, 1.0)},
            {"param": (np.zeros(1), np.eye(1))},
        ]
        return params
