"""Tests for regression data generation"""

import numpy as np
import pytest

from skchange.new_api.datasets import generate_piecewise_regression_data


def test_generate_piecewise_regression_data_default():
    """Test default generation of piecewise regression data."""
    X, y = generate_piecewise_regression_data()
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]


@pytest.mark.parametrize("lengths", [100, [100], [50, 50], [30, 20, 50]])
def test_generate_piecewise_regression_data_valid_lengths(lengths):
    X, y, params = generate_piecewise_regression_data(
        lengths=lengths, return_params=True
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0]
    assert isinstance(params, dict)


@pytest.mark.parametrize("lengths", [[], -10, [100, -50]])
def test_generate_piecewise_regression_data_invalid_lengths(lengths):
    with pytest.raises(ValueError):
        generate_piecewise_regression_data(lengths=lengths)
