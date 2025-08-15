"""Examples of conditional fixtures for skchange testing."""

import numpy as np
import pandas as pd
import pytest

from skchange.costs import L2Cost, PoissonCost
from skchange.datasets import generate_alternating_data, generate_anomalous_data


@pytest.fixture(params=[L2Cost, PoissonCost])
def scorer_with_appropriate_data(request):
    """Provide scorer with data appropriate for its type.

    This demonstrates how request.param works with classes.
    """
    ScorerClass = request.param  # Gets L2Cost, then PoissonCost
    scorer = ScorerClass.create_test_instance()

    # Generate data based on scorer requirements
    distribution_type = scorer.get_tag("distribution_type", None)

    if distribution_type == "Poisson":
        # Generate integer data for Poisson
        data = np.random.poisson(5, size=(100, 3))
        df = pd.DataFrame(data)
    else:
        # Generate normal data for other scorers
        df = generate_anomalous_data()

    return scorer, df


@pytest.fixture(
    params=[
        {"n_segments": 1, "segment_length": 50},
        {"n_segments": 3, "segment_length": 30},
        {"n_segments": 5, "segment_length": 20},
    ]
)
def test_scenarios(request):
    """Parametrized fixture with dictionary parameters.

    Shows how request.param works with complex objects.
    """
    params = request.param  # Gets each dictionary

    # Use the parameters to generate data
    data = generate_alternating_data(
        n_segments=params["n_segments"],
        segment_length=params["segment_length"],
        p=2,
        random_state=42,
    )

    # Return both the parameters and the data
    return {
        "data": data,
        "expected_changepoints": params["n_segments"] - 1,
        "total_length": params["n_segments"] * params["segment_length"],
    }


@pytest.fixture(
    params=[
        pytest.param(1, id="single_dimension"),
        pytest.param(3, id="three_dimensions"),
        pytest.param(5, id="five_dimensions"),
    ]
)
def dimensional_data(request):
    """Parametrized fixture with custom test IDs.

    The id parameter makes test output more readable.
    """
    n_dims = request.param

    # Generate data with specified dimensions
    data = np.random.randn(100, n_dims)
    df = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(n_dims)])

    return df


# Conditional fixture based on test function name
@pytest.fixture
def adaptive_data(request):
    """Provide different data based on the test function name.

    This shows how to use request.function for dynamic behavior.
    """
    test_name = request.function.__name__

    if "anomaly" in test_name:
        return generate_anomalous_data()
    elif "alternating" in test_name:
        return generate_alternating_data(n_segments=2, segment_length=50, p=3)
    else:
        # Default data
        return pd.DataFrame(np.random.randn(100, 3))


# Indirect parametrization - more advanced
@pytest.fixture
def scorer_instance(request):
    """Create scorer instance from class passed via indirect parametrization."""
    ScorerClass = request.param
    return ScorerClass.create_test_instance()


# Example tests using these fixtures
def test_scorer_with_data(scorer_with_appropriate_data):
    """Test scorer with appropriate data type."""
    scorer, data = scorer_with_appropriate_data
    scorer.fit(data)
    assert scorer.is_fitted


def test_different_scenarios(test_scenarios):
    """Test with different data scenarios."""
    scenario = test_scenarios
    data = scenario["data"]

    assert len(data) == scenario["total_length"]
    # Additional assertions based on scenario parameters


def test_dimensional_fitting(dimensional_data):
    """Test fitting with different dimensions."""
    data = dimensional_data
    n_features = data.shape[1]

    # Test that data has expected structure
    assert data.shape[0] == 100
    assert data.shape[1] == n_features


def test_anomaly_detection(adaptive_data):
    """Test that includes 'anomaly' in name - gets anomalous data."""
    data = adaptive_data
    # This test automatically gets anomalous data due to its name
    assert isinstance(data, pd.DataFrame)


def test_alternating_patterns(adaptive_data):
    """Test that includes 'alternating' in name - gets alternating data."""
    data = adaptive_data
    # This test automatically gets alternating data due to its name
    assert isinstance(data, pd.DataFrame)


# Using indirect parametrization (more advanced)
@pytest.mark.parametrize(
    "scorer_instance",
    [L2Cost, PoissonCost],
    indirect=True,  # Tells pytest to pass these to the fixture
)
def test_indirect_scorer(scorer_instance):
    """Test using indirect parametrization."""
    scorer = scorer_instance
    assert hasattr(scorer, "fit")
    assert hasattr(scorer, "evaluate")
