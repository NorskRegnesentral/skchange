"""Tests for LinearRegressionCost class."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from skchange.costs.linear_regression_cost import LinearRegressionCost


def test_linear_regression_cost_init():
    """Test initialization of LinearRegressionCost."""
    # Default parameters
    cost = LinearRegressionCost()
    assert cost.response_col == 0

    # Custom response_col
    cost = LinearRegressionCost(response_col=2)
    assert cost.response_col == 2

    # Invalid response_col type
    with pytest.raises(ValueError):
        LinearRegressionCost(response_col="invalid")


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
        cost = LinearRegressionCost()
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
    lr = LinearRegression()
    lr.fit(X, y)
    expected_cost = np.sum((lr.predict(X) - y) ** 2)

    # Allow for small numerical differences
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_linear_regression_cost_evaluate_multiple_intervals():
    """Test evaluation of LinearRegressionCost on multiple intervals."""
    # Create data
    n_samples = 200
    X = np.random.rand(n_samples, 3)

    # Fit the cost
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
    cost = LinearRegressionCost()
    # Number of parameters is equal to number of variables
    assert cost.get_param_size(5) == 4
