"""Linear Regression cost function.

This module contains the LinearRegressionCost class, which is a cost function for
change point detection based on linear regression. The cost is the sum of squared
residuals from fitting a linear regression model within each segment.
"""

import numpy as np

from skchange.costs import BaseCost
from skchange.utils.numba import njit
from skchange.utils.validation.enums import EvaluationType


@njit
def linear_regression_cost(X: np.ndarray, y: np.ndarray) -> float:
    """Compute the cost for a linear regression model.

    Parameters
    ----------
    X : np.ndarray
        Features.
    y : np.ndarray
        Target values.

    Returns
    -------
    cost : float
        Sum of squared residuals from the linear regression.
    """
    # Returns: (coeffs, residuals, X_rank, X_singular_values):
    _, residuals, _, _ = np.linalg.lstsq(X, y)
    if residuals.size == 0:
        # Underdetermined system, residuals are zero.
        # If rank(X) < X.shape[1], or X.shape[0] <= X.shape[1],
        # "residuals" is an empty array.
        return 0.0
    else:
        # If y is 1-dimensional, this is a (1,) shape array.
        # Otherwise the shape is (y.shape[1],).
        return np.sum(residuals)


@njit
def linear_regression_cost_intervals(
    starts: np.ndarray, ends: np.ndarray, X: np.ndarray, response_col: int
) -> np.ndarray:
    """Evaluate the linear regression cost for each interval.

    Parameters
    ----------
    starts : np.ndarray
        Start indices of the intervals (inclusive).
    ends : np.ndarray
        End indices of the intervals (exclusive).
    X : np.ndarray
        Data to evaluate. Must be a 2D array.
    response_col : int
        Index of column in X to use as the response variable.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs with one row for each interval and one column.
    """
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        segment = X[start:end]

        if segment.shape[0] <= 1:  # Not enough data points
            costs[i, 0] = 0.0
            continue

        y = segment[:, response_col : response_col + 1]  # Keep as 2D array

        # Create feature matrix without the response column
        if response_col > 0:
            X_features = np.hstack(
                (segment[:, :response_col], segment[:, response_col + 1 :])
            )
        else:
            X_features = segment[:, 1:]

        # Add a column of ones for the intercept
        X_features = np.hstack((np.ones((X_features.shape[0], 1)), X_features))

        # Compute cost for this interval
        costs[i, 0] = linear_regression_cost(X_features, y)

    return costs


class LinearRegressionCost(BaseCost):
    """Linear Regression sum of squared residuals cost.

    This cost computes the sum of squared residuals from fitting a linear
    regression model within each segment. One column of the input data X is
    used as the response variable, and the remaining columns are used as
    predictors.

    Parameters
    ----------
    param : any, optional (default=None)
        Not used but kept for API compatibility.
    response_col : int, optional (default=0)
        Index of column in X to use as the response variable.
    """

    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": "Tveten",
    }

    evaluation_type = EvaluationType.MULTIVARIATE
    supports_fixed_params = False

    def __init__(
        self,
        param=None,
        response_col=0,
    ):
        super().__init__(param)

        if not isinstance(response_col, int):
            raise ValueError("response_col must be an integer")

        self.response_col = response_col

    def _fit(self, X: np.ndarray, y=None):
        """Fit the cost.

        This method validates input data and stores it for cost evaluation.

        Parameters
        ----------
        X : np.ndarray
            Data to evaluate. Must be a 2D array.
        y: None
            Ignored. Included for API consistency by convention.
        """
        # Check that X has enough columns
        if X.shape[1] <= 1:
            raise ValueError(
                "X must have at least 2 columns for linear regression "
                "(1 for response and at least 1 for predictors)."
            )

        # Check that response_col is valid
        if not 0 <= self.response_col < X.shape[1]:
            raise ValueError(
                f"response_col ({self.response_col}) must be between 0 and {X.shape[1] - 1}"
            )

        self._param = self._check_param(self.param, X)

        return self

    def _evaluate_optim_param(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        """Evaluate the linear regression cost for each interval.

        Parameters
        ----------
        starts : np.ndarray
            Start indices of the intervals (inclusive).
        ends : np.ndarray
            End indices of the intervals (exclusive).

        Returns
        -------
        costs : np.ndarray
            A 2D array of costs with one row for each interval and one column.
        """
        return linear_regression_cost_intervals(
            starts, ends, self._X, self.response_col
        )

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        if self.is_fitted:
            # Need at least as many samples as features to have a determined system.
            # Subtract 1 for the response variable.
            return self._X.shape[1] - 1
        else:
            return None

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
        # Number of parameters = all features except response variable
        return p - 1

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
            {"response_col": 0},
            {"response_col": 1},
        ]
        return params
