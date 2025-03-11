"""Tests for LinearRegressionCost class."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from skchange.costs.linear_regression_cost import LinearRegressionCost


def test_linear_regression_cost_init():
    """Test initialization of LinearRegressionCost."""
    # Default parameters
    cost = LinearRegressionCost(response_col=0)
    assert cost.response_col == 0

    # Custom response_col
    cost = LinearRegressionCost(response_col=2)
    assert cost.response_col == 2

    # Invalid response_col type:
    with pytest.raises(ValueError):
        LinearRegressionCost(response_col="invalid").fit(np.random.rand(100, 3))


def test_linear_regression_cost_fit():
    """Test fitting of LinearRegressionCost."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Valid fit
    cost = LinearRegressionCost(response_col=1)
    cost.fit(X)
    assert cost.is_fitted

    # Invalid number of columns
    X_single_col = np.random.rand(100, 1)
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(response_col=0)
        cost.fit(X_single_col)

    # Invalid response_col
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(response_col=5)  # Out of bounds
        cost.fit(X)


def test_linear_regression_cost_evaluate():
    """Test evaluation of LinearRegressionCost."""
    # Create regression dataset with known relationship
    X, y = make_regression(
        n_samples=100, n_features=3, n_informative=3, noise=0.1, random_state=42
    )
    # Add y as last column to X
    X_with_y = np.hstack((X, y.reshape(-1, 1)))

    # Fit the cost with y as the response (last column)
    cost = LinearRegressionCost(response_col=3)
    cost.fit(X_with_y)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([100])
    cuts = np.hstack((starts, ends))
    costs = cost.evaluate(cuts=cuts)

    # Compare with sklearn's LinearRegression
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)
    expected_cost = np.sum((lr.predict(X) - y) ** 2)

    # Allow for small numerical differences:
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_linear_regression_cost_evaluate_multiple_intervals():
    """Test evaluation of LinearRegressionCost on multiple intervals."""
    # Create data
    n_samples = 200
    X = np.random.rand(n_samples, 3)

    # Fit the cost:
    cost = LinearRegressionCost(response_col=0)
    cost.fit(X)

    # Define intervals
    starts = np.array([0, 50, 100, 150])
    ends = np.array([50, 100, 150, 200])
    cuts = np.hstack([starts.reshape(-1, 1), ends.reshape(-1, 1)])

    # Evaluate
    costs = cost.evaluate(cuts=cuts)

    # Check shape
    assert costs.shape == (4, 1)

    # Check all costs are non-negative
    assert np.all(costs >= 0)


def test_min_size_property():
    """Test the min_size property."""
    # Before fitting
    cost = LinearRegressionCost(response_col=0)
    assert cost.min_size is None

    # After fitting with 3 columns (response + 2 features)
    X = np.random.rand(100, 3)
    cost.fit(X)
    assert cost.min_size == 2  # 2 features


def test_get_param_size():
    """Test get_param_size method."""
    cost = LinearRegressionCost(response_col="log_house_price")
    # Number of parameters is equal to number of variables
    assert cost.get_param_size(5) == 4


def test_simple_linear_regression_cost_fixed_params():
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=np.float64)

    # y = 1 * x_0 + 2 * x_1 + 3:
    y: np.ndarray = np.dot(X, np.array([1, 2])) + 3.0

    reg = LinearRegression(fit_intercept=False).fit(X, y)

    fixed_coef = reg.coef_

    cost = LinearRegressionCost(param=fixed_coef, response_col=0)
    cost.fit(np.hstack((y.reshape(-1, 1), X)))

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([4])
    cuts = np.hstack((starts, ends))
    costs = cost.evaluate(cuts=cuts)

    # Calculate expected cost manually:
    y_pred = reg.predict(X)
    expected_cost = np.sum(np.square(y - y_pred))

    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_linear_regression_cost_fixed_params():
    """Test evaluation with fixed parameters."""
    # Create regression dataset
    X, y = make_regression(
        n_samples=10, n_features=2, n_informative=2, noise=0.1, random_state=42
    )
    X_with_y = np.hstack((X, y.reshape(-1, 1)))

    # First fit a regular linear regression to get coefficients
    lr = LinearRegression(fit_intercept=False)
    lr.fit(X, y)

    # Create coefficient array (intercept + coeffs)
    fixed_coeffs = lr.coef_

    # Create cost with fixed params
    cost = LinearRegressionCost(param=fixed_coeffs, response_col=2)
    cost.fit(X_with_y)

    # Evaluate on the entire interval
    starts = np.array([0])
    ends = np.array([100])
    cuts = np.hstack((starts.reshape(-1, 1), ends.reshape(-1, 1)))
    costs = cost.evaluate(cuts=cuts)

    # Calculate expected cost manually
    y_pred = lr.predict(X)
    expected_cost = np.sum(np.square(y - y_pred))

    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_fixed_param_validation():
    """Test validation of fixed parameters."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # Valid parameters: (2 features excluding response)
    valid_params = np.array([0.2, -0.3])
    cost = LinearRegressionCost(param=valid_params, response_col=0)
    cost.fit(X)
    assert np.array_equal(cost._coeffs, valid_params.reshape(-1, 1))

    # Valid column vector params:
    valid_params_col = np.array([[0.2], [-0.3]])
    cost = LinearRegressionCost(param=valid_params_col, response_col=0)
    cost.fit(X)
    assert np.array_equal(cost._coeffs, valid_params_col)

    # Invalid parameter dimension (1, 2):
    invalid_params_2d = np.array([[0.2, 0.3]])
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(param=invalid_params_2d, response_col=0)
        cost.fit(X)

    # Invalid parameter length
    invalid_params_length = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        cost = LinearRegressionCost(param=invalid_params_length, response_col=0)
        cost.fit(X)


def test_min_size_with_fixed_params():
    """Test min_size property with fixed parameters."""
    # Create simple dataset
    X = np.random.rand(100, 3)

    # With optimized parameters
    cost = LinearRegressionCost(response_col=1)
    assert cost.min_size is None  # Not fitted yet
    cost.fit(X)
    assert cost.min_size == 2  # Need 2 samples for 2 parameters

    # With fixed parameters
    fixed_params = np.array([0.1, 0.2])
    cost_fixed = LinearRegressionCost(param=fixed_params, response_col=1)
    cost_fixed.fit(X)
    assert cost_fixed.min_size == 1  # Only need 1 sample with fixed parameters

