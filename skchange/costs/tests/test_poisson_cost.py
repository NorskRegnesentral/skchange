import numpy as np
import pytest
import scipy.stats as stats

from skchange.costs.poisson_cost import (
    PoissonCost,
    fast_poisson_log_likelihood,
    fast_poisson_mle_rate_log_likelihood,
)


def test_poisson_log_likelihood():
    """Test that fast_poisson_log_likelihood agrees with scipy implementation."""
    # Generate random Poisson samples with known rate
    rate = 3.5
    poisson_sample = stats.poisson.rvs(rate, size=1000)

    # Calculate log-likelihood using our fast implementation
    fast_ll = fast_poisson_log_likelihood(rate, poisson_sample)

    # Calculate log-likelihood using scipy
    scipy_ll = stats.poisson.logpmf(poisson_sample, rate).sum()

    # Check that they are close
    assert np.isclose(fast_ll, scipy_ll, rtol=1e-10)


def test_poisson_mle_log_likelihood():
    """Test that log likelihood evaluation agrees with scipy implementation."""
    # Generate random Poisson samples
    np.random.seed(42)
    true_rate = 5.0
    poisson_sample = stats.poisson.rvs(true_rate, size=1000)

    # Calculate MLE rate (sample mean)
    mle_rate = np.mean(poisson_sample)

    # Calculate log-likelihood using our fast implementation with MLE rate
    fast_ll = fast_poisson_mle_rate_log_likelihood(mle_rate, poisson_sample)

    # Calculate log-likelihood using scipy with MLE rate
    scipy_ll = stats.poisson.logpmf(poisson_sample, mle_rate).sum()

    # Check that they are close
    assert np.isclose(fast_ll, scipy_ll, rtol=1e-10)


def test_poisson_cost_fixed_param():
    """Test PoissonCost with fixed parameter."""
    # Generate Poisson data
    np.random.seed(42)
    rate = 4.0
    n_samples = 100
    poisson_data = stats.poisson.rvs(rate, size=(n_samples, 1))

    # Create PoissonCost with fixed rate
    cost = PoissonCost(param=rate)
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_cost = -2 * stats.poisson.logpmf(poisson_data, rate).sum()

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_poisson_cost_mle_param():
    """Test PoissonCost with MLE parameter."""
    # Generate Poisson data
    np.random.seed(42)
    true_rate = 4.0
    n_samples = 100
    poisson_data = stats.poisson.rvs(true_rate, size=(n_samples, 1))

    # Create PoissonCost with MLE rate
    cost = PoissonCost()
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate MLE rate
    mle_rate = np.mean(poisson_data)

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_cost = -2 * stats.poisson.logpmf(poisson_data, mle_rate).sum()

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_cost, rtol=1e-10)


def test_poisson_cost_input_validation():
    """Test input validation in PoissonCost."""
    # Try to fit with non-integer data
    with pytest.raises(ValueError):
        cost = PoissonCost()
        cost.fit(np.array([[1.5, 2.3], [3.1, 4.2]]))

    # Try to fit with negative integer data
    with pytest.raises(ValueError):
        cost = PoissonCost()
        cost.fit(np.array([[1, 2], [-3, 4]], dtype=int))

    # Try to create with negative rate parameter
    with pytest.raises(ValueError):
        cost = PoissonCost(param=-1.0)
        cost.fit(np.array([[1, 2], [3, 4]], dtype=int))


def test_poisson_cost_on_two_columns():
    """Test PoissonCost on two columns."""
    # Generate Poisson data
    np.random.seed(42)
    rate1 = 4.0
    rate2 = 2.0
    n_samples = 100
    poisson_data = np.column_stack(
        (
            stats.poisson.rvs(rate1, size=n_samples),
            stats.poisson.rvs(rate2, size=n_samples),
        )
    )

    # Create PoissonCost with fixed rate
    cost = PoissonCost(param=[rate1, rate2])
    cost.fit(poisson_data)

    # Evaluate on the whole dataset
    costs = cost.evaluate(np.array([[0, n_samples]]))

    # Calculate expected cost using scipy (twice negative log-likelihood)
    expected_costs = [
        -2.0 * stats.poisson.logpmf(poisson_data[:, 0], rate1).sum(),
        -2.0 * stats.poisson.logpmf(poisson_data[:, 1], rate2).sum(),
    ]

    # Check that they are close
    assert np.isclose(costs[0, 0], expected_costs[0], rtol=1e-10)
    assert np.isclose(costs[0, 1], expected_costs[1], rtol=1e-10)
