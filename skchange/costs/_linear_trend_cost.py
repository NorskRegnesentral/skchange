"""Linear trend cost function.

This module contains the LinearTrendCost class, which is a cost function for
change point detection based on the squared error between data points
and a fitted linear trend line within each interval.
"""

import numpy as np

from ..utils.numba import njit
from ..utils.validation.enums import EvaluationType
from .base import BaseCost

# todo: add the cost to the COSTS variable in skchange.costs.__init__.py


@njit
def fit_linear_trend(time_steps: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    """Calculate the optimal linear trend for a given array.

    Parameters
    ----------
    x : np.ndarray
        1D array of data points

    Returns
    -------
    tuple
        (slope, intercept) of the best-fit line
    """
    n = len(values)
    if n < 3:
        return 0.0, 0.0

    # Use the fact that for evenly spaced x values starting at 0,
    # the slope and intercept can be calculated more efficiently
    mean_t = np.mean(time_steps)
    centered_time_steps = time_steps - mean_t
    mean_value = np.mean(values)

    # Calculate linear regression denominator = sum((t-mean_t)²):
    denominator = np.sum(centered_time_steps**2)

    # Calculate linear regression numerator = sum((t-mean_t)*(x-mean_x)):
    numerator = np.sum(centered_time_steps * (values - mean_value))

    slope = numerator / denominator
    intercept = mean_value - slope * mean_t

    return slope, intercept


@njit
def linear_trend_cost(
    xs: np.ndarray, ts: np.ndarray, slope: float, intercept: float
) -> float:
    """Calculate the squared error cost between array and a linear trend.

    Parameters
    ----------
    xs : np.ndarray
        1D array of data points
    ts : np.ndarray
        1D array of time steps corresponding to the data points
    slope : float
        Slope of the linear trend
    intercept : float
        Intercept of the linear trend

    Returns
    -------
    float
        Sum of squared errors between data points and the linear trend
    """
    linear_trend = intercept + slope * ts
    return np.sum((xs - linear_trend) ** 2)


@njit
def linear_trend_cost_mle(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, ts: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with optimized parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    ts : np.ndarray
        Time steps for the data points. By default, this is the index of the data.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        for col in range(n_columns):
            segment_data = X[start:end, col]
            segment_ts = ts[start:end]

            slope, intercept = fit_linear_trend(segment_data, segment_ts)
            costs[i, col] = linear_trend_cost(
                segment_data, segment_ts, slope, intercept
            )

    return costs


@njit
def linear_trend_cost_fixed(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Evaluate the linear trend cost with fixed parameters.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    params : np.ndarray
        Fixed parameters for linear trend: [[slope_1, intercept_1],
                                            [slope_2, intercept_2],
                                             ...]
        where each pair corresponds to a column in X.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs. One row for each interval. The number of columns
        is equal to the number of columns in the input data.
    """
    n_intervals = len(starts)
    n_columns = X.shape[1]
    costs = np.zeros((n_intervals, n_columns))

    for col in range(n_columns):
        slope, intercept = params[col, :]
        for i in range(n_intervals):
            start, end = starts[i], ends[i]
            segment = X[start:end, col]
            costs[i, col] = linear_trend_cost(segment, slope, intercept)

    return costs


class LinearTrendCost(BaseCost):
    """Linear trend cost function.

    This cost function calculates the sum of squared errors between data points
    and a linear trend line fitted to each interval. For each interval and each column,
    a straight line is fitted (or provided as a fixed parameter) and the squared
    differences between the actual data points and the fitted line are summed.

    Parameters
    ----------
    param : array-like, optional (default=None)
        Fixed parameters for the cost calculation in the form [slope_1, intercept_1, slope_2, intercept_2, ...]
        where each pair corresponds to a column in the input data.
        If None, the optimal parameters (i.e., the best-fit line) are calculated for each interval.
    """

    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": "johannvk",
    }

    evaluation_type = EvaluationType.UNIVARIATE
    supports_fixed_params = True

    def __init__(self, param=None):
        super().__init__(param)
        self._fixed_params = None

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method stores the input data for later cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        self._param = self._check_param(self.param, X)
        if self.param is not None:
            self._fixed_params = self._param

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the cost for the optimal linear trend parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        For each interval, a straight line is fitted and the squared differences
        between the actual data points and the fitted line are summed.

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
            columns is equal to the number of columns in the input data.
        """
        return linear_trend_cost_mle(starts, ends, self._X)

    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for fixed linear trend parameters.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.
        For each interval, the squared differences between the actual data points and
        the fixed linear trend are summed.

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
            columns is equal to the number of columns in the input data.
        """
        return linear_trend_cost_fixed(starts, ends, self._X, self._fixed_params)

    def _check_fixed_param(self, param, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : array-like
            Fixed parameters for the linear trend: [[slope_1, intercept_1],
                                                    [slope_2, intercept_2],
                                                     ...]
            where each pair of parameters corresponds to a column in X.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: np.ndarray
            Fixed parameters for the cost calculation.
        """
        if param is None:
            return param

        # Convert to numpy array if not already
        param_array = np.asarray(param, dtype=float)

        # Check that we have the right number of parameters (2 per column)
        if param_array.size != 2 * X.shape[1]:
            raise ValueError(
                f"Expected {2 * X.shape[1]} parameters "
                f"(2 per column), but got {param_array.size}."
            )

        if param_array.ndim == 1 and X.shape[1] == 1:
            # Got a 1D array for a single column, reshape to 2D.
            param_array = param_array.reshape(-1, 2)

        if param_array.ndim != 2:
            raise ValueError("Fixed parameters must be convertible to a 2D array.")

        return param_array

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        The size of each interval is defined as ``cuts[i, 1] - cuts[i, 0]``.
        For linear trend fitting, we need at least 3 points.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        return 3

    def get_param_size(self, p: int) -> int:
        """Get the number of parameters in the cost function.

        Parameters
        ----------
        p : int
            Number of variables in the data.

        Returns
        -------
        int
            Number of parameters in the cost function.
        """
        # For each column we need 2 parameters: slope and intercept
        return 2 * p

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests.

        Returns
        -------
        params : list of dict
            Parameters to create testing instances of the class
        """
        params = [
            {"param": None},
            {"param": np.array([1.0, 0.0])},  # slope=1, intercept=0 for a single column
        ]
        return params
