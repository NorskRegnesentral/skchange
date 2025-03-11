"""Linear Regression cost function.

This module contains the LinearRegressionCost class, which is a cost function for
change point detection based on linear regression. The cost is the sum of squared
residuals from fitting a linear regression model within each segment.
"""

import numpy as np
import pandas as pd
from sktime.utils.validation.series import check_series

from skchange.costs import BaseCost
from skchange.utils.numba import njit
from skchange.utils.validation.data import as_2d_array
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
def linear_regression_cost_fixed_params(
    starts: np.ndarray,
    ends: np.ndarray,
    response_data: np.ndarray,
    covariate_data: np.ndarray,
    coeffs: np.ndarray,
) -> np.ndarray:
    """Evaluate the linear regression cost for fixed parameters.

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
    coeffs : np.ndarray (shape (K, 1))
        Fixed regression coefficients. First element is the intercept.

    Returns
    -------
    costs : np.ndarray
        A 2D array of costs with one row for each interval and one column.
    """
    n_intervals = len(starts)
    costs = np.zeros((n_intervals, 1))

    for i in range(n_intervals):
        start, end = starts[i], ends[i]
        X_features = covariate_data[start:end]
        y_response = response_data[start:end, None]

        # Compute predictions using fixed parameters:
        y_pred = X_features @ coeffs

        # Calculate residual sum of squares:
        costs[i, 0] = np.sum(np.square(y_response - y_pred))

    return costs


@njit
def linear_regression_cost_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    response_data: np.ndarray,
    covariate_data: np.ndarray,
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
        X_features = covariate_data[start:end]
        y_response = response_data[start:end]
        costs[i, 0] = linear_regression_cost(X_features, y_response)

    return costs


def check_data_column(
    data_column: int | str,
    column_role: str,
    X: np.ndarray,
    X_columns: pd.Index | None,
) -> int:
    """Check that a data column name or index is valid.

    Parameters
    ----------
    data_column : int or str
        Column index or name to check.
    column_role : str
        Role of the column (e.g., "Response").
    X : np.ndarray
        Data array.
    X_columns : pd.Index or None
        Column names of the data array.

    Returns
    -------
    data_column : int
        Column index.
    """
    if isinstance(data_column, int):
        if not 0 <= data_column < X.shape[1]:
            raise ValueError(
                f"{column_role} column index ({data_column}) must"
                f" be between 0 and {X.shape[1] - 1}."
            )
    elif isinstance(data_column, str) and X_columns is not None:
        if data_column not in X_columns:
            raise ValueError(
                f"{column_role} column ({data_column}) not found "
                f"among the fit data columns: {X_columns}."
            )
        data_column = X_columns.get_loc(data_column)
    else:
        raise ValueError(
            f"{column_role} column must be an integer in the range "
            f"[0, {X.shape[1]}), or a valid column name. Got {data_column}."
        )
    return data_column


class LinearRegressionCost(BaseCost):
    """Linear Regression sum of squared residuals cost.

    This cost computes the sum of squared residuals from fitting a linear
    regression model within each segment. One column of the input data X is
    used as the response variable, and the remaining columns are used as
    predictors.

    Parameters
    ----------
    param : array-like, optional (default=None)
        Fixed regression coefficients. If None, coefficients are estimated
        for each interval using ordinary least squares. If provided, must be an array
        where the first element is the intercept term, followed by coefficients
        for each predictor variable.
    response_col : int, optional (default=0)
        Index of column in X to use as the response variable.
    """

    _tags = {
        "authors": ["Tveten", "johannvk"],
        "maintainers": "Tveten",
    }

    evaluation_type = EvaluationType.MULTIVARIATE
    supports_fixed_params = True

    def __init__(
        self,
        response_col: str | int,
        covariate_cols: list[str | int] = None,
        param=None,
    ):
        super().__init__(param)

        self.response_col = response_col
        self.covariate_cols = covariate_cols

        self._response_col_idx: int | None = None
        self._covariate_col_indices: list[int] | None = None

        self._response_data: np.ndarray | None = None
        self._covariate_data: np.ndarray | None = None

        self._coeffs: np.ndarray | None = None

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

        self._param = self._check_param(self.param, X)
        if self.param is not None:
            self._coeffs = self._param

        # Check that response_col is valid:
        self._response_col_idx = check_data_column(
            self.response_col, "Response", X, self._X_columns
        )
        if self.covariate_cols is not None:
            self._covariate_col_indices = [
                check_data_column(col, "Covariate", X, self._X_columns)
                for col in self.covariate_cols
            ]
        else:
            # Use all columns except the response column as covariates:
            self._covariate_col_indices = list(range(X.shape[1]))
            self._covariate_col_indices.remove(self._response_col_idx)

        self._response_data = X[:, self.response_col]
        self._covariate_data = X[:, self._covariate_col_indices]

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
            starts, ends, self._response_data, self._covariate_data
        )

    def _evaluate_fixed_param(self, starts, ends) -> np.ndarray:
        """Evaluate the cost for fixed regression coefficients.

        Evaluates the cost for `X[start:end]` for each each start, end in starts, ends.

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
        return linear_regression_cost_fixed_params(
            starts, ends, self._response_data, self._covariate_data, self._coeffs
        )

    def _check_fixed_param(self, param, X: np.ndarray) -> np.ndarray:
        """Check if the fixed parameter is valid relative to the data.

        Parameters
        ----------
        param : array-like
            Fixed regression coefficients.
        X : np.ndarray
            Input data.

        Returns
        -------
        param: np.ndarray
            Fixed regression coefficients for cost calculation.
        """
        param = np.asarray(param)

        # Expected number of coefficients: predictors (excluding response)
        expected_length = X.shape[1] - 1

        if len(param) != expected_length:
            raise ValueError(
                f"Expected {expected_length} coefficients"
                f" ({expected_length} predictors), got {len(param)}"
            )

        if param.ndim != 1 and param.shape[1] != 1:
            raise ValueError(
                f"Coefficients must have shape ({expected_length}, 1)  or"
                f" ({expected_length},). Got shape {param.shape}."
            )

        return param.reshape(-1, 1)

    @property
    def min_size(self) -> int:
        """Minimum size of the interval to evaluate.

        Returns
        -------
        int
            The minimum valid size of an interval to evaluate.
        """
        # For fixed parameter evaluation, we only need a single sample:
        if self.param is not None:
            return 1

        if self.is_fitted:
            # Need at least as many samples as covariates:
            return len(self._covariate_col_indices)
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
        # Number of parameters = all features except response variable.
        if self.is_fitted:
            # Could use fewer covariates than the total number of columns:
            return len(self._covariate_col_indices)
        else:
            # Default to all columns except the response variable:
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
            {"param": None, "response_col": 0},
            {"param": np.array([1.0, 0.5, -0.3]), "response_col": 1},
        ]
        return params
