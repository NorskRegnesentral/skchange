"""Shared fixtures and test utilities for skchange tests."""

import numpy as np
import pandas as pd
import pytest

from skchange.datasets import generate_alternating_data, generate_anomalous_data


@pytest.fixture
def sample_anomalous_data():
    """Generate sample anomalous data with datetime index."""
    x = generate_anomalous_data()
    x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")
    return x


@pytest.fixture
def sample_alternating_data():
    """Generate sample alternating data for testing."""
    return generate_alternating_data(
        n_segments=2,
        mean=20,
        segment_length=50,
        p=3,
        random_state=15,
    )


@pytest.fixture
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    yield
    # Reset random state after test
    np.random.seed(None)


@pytest.fixture(params=[1, 3, 5])
def different_dimensions(request):
    """Parametrized fixture for testing different data dimensions."""
    return request.param


@pytest.fixture(
    params=[
        pytest.param("small", id="small_dataset"),
        pytest.param("medium", id="medium_dataset"),
        pytest.param("large", id="large_dataset"),
    ]
)
def dataset_size(request):
    """Parametrized fixture for different dataset sizes."""
    size_config = {
        "small": {"n_samples": 50, "n_features": 2},
        "medium": {"n_samples": 200, "n_features": 5},
        "large": {"n_samples": 1000, "n_features": 10},
    }
    return size_config[request.param]


@pytest.fixture
def scorer_with_compatible_data(request):
    """Conditional fixture that provides scorer with appropriate test data.

    Uses request to access the current test's information and provide
    compatible data based on scorer requirements.
    """
    # This would be used with indirect parametrization
    # where request.param contains a scorer class
    if hasattr(request, "param"):
        ScorerClass = request.param
        scorer = ScorerClass.create_test_instance()

        # Generate appropriate data based on scorer tags
        distribution_type = scorer.get_tag("distribution_type", None)
        is_conditional = scorer.get_tag("is_conditional", False)

        if distribution_type == "Poisson":
            # Generate Poisson data
            data = np.random.poisson(5, size=(100, 3))
            x = pd.DataFrame(data, columns=[f"feature_{i}" for i in range(3)])
        elif is_conditional:
            # Skip if conditional scorer (no test data yet)
            pytest.skip(f"No test data for conditional scorer {ScorerClass.__name__}")
        else:
            # Use standard anomalous data
            x = generate_anomalous_data()
            x.index = pd.date_range(start="2020-01-01", periods=x.shape[0], freq="D")

        return scorer, x
    else:
        # Fallback if no parameter provided
        return None, generate_anomalous_data()


@pytest.fixture
def sample_cuts():
    """Generate sample cuts for testing."""
    return {
        "single_cut": np.array([0, 40]),
        "multiple_cuts": np.array([[0, 20], [20, 40]]),
        "invalid_float_cut": np.array([0.5, 40.5]),
        "invalid_reversed_cut": np.array([40, 0]),
    }
