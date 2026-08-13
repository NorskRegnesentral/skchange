"""Tests for continous piecewise linear data generation."""

import numpy as np
import pytest

from skchange.new_api.datasets import generate_continuous_piecewise_linear_data


def test_generate_continuous_piecewise_linear_data_default():
    arr = generate_continuous_piecewise_linear_data()
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    assert arr.size > 0


@pytest.mark.parametrize(
    "slopes",
    [
        None,
        1,
        [1],
        [2, 3],
        [0.5, -0.5, 1],
    ],
)
def test_generate_continuous_piecewise_linear_data_valid_slopes(
    slopes: float | list[float],
):
    """Test that invalid slopes raise ValueError."""
    arr = generate_continuous_piecewise_linear_data(slopes=slopes)
    assert arr.size > 0


def test_generate_continuous_piecewise_linear_data_invalid_noise_std():
    """Test that invalid noise_std raises ValueError."""
    with pytest.raises(ValueError):
        generate_continuous_piecewise_linear_data(noise_std=-1)
