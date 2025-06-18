"""Tests for the generate_piecewise_normal_data function."""

import numpy as np
import pytest

from skchange.datasets import generate_piecewise_normal_data


def test_generate_piecewise_normal_data_default_params():
    """Test that the function generates data with default parameters."""
    df = generate_piecewise_normal_data(10, 2)
    assert df.shape == (10, 2)


@pytest.mark.parametrize("n", [1, 2, 10, 100])
def test_generate_piecewise_normal_data_valid_n(n: int):
    """Test that the function generates data with the correct number of rows."""
    df = generate_piecewise_normal_data(n=n, random_state=43)
    assert df.shape[0] == n


@pytest.mark.parametrize("n", [-1, 0, 0.5])
def test_generate_piecewise_normal_data_invalid_n(n: int):
    """Test that the function raises ValueError for invalid n values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n=n, random_state=43)


@pytest.mark.parametrize("p", [1, 2, 10, 100])
def test_generate_piecewise_normal_data_valid_p(p: int):
    """Test that the function generates data with the correct number of columns."""
    df = generate_piecewise_normal_data(n=10, p=p)
    assert df.shape[1] == p


@pytest.mark.parametrize("p", [-1, 0, 0.5])
def test_generate_piecewise_normal_data_invalid_p(p: int):
    """Test that the function raises ValueError for invalid p values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n=10, p=p)


@pytest.mark.parametrize("n_change_points", [0, 1, 2, 9, None])
def test_generate_piecewise_normal_data_n_cpts(n_change_points: int):
    """Test that the function generates data with correct number of change points."""
    df, params = generate_piecewise_normal_data(
        n=10, n_change_points=n_change_points, return_params=True
    )
    change_points = params["change_points"]
    if n_change_points is not None:
        assert len(change_points) == n_change_points
    else:
        assert len(change_points) >= 0


@pytest.mark.parametrize("n_change_points", [-1, 10])
def test_generate_piecewise_normal_data_invalid_n_cpts(n_change_points: int):
    """Test that the function raises ValueError for invalid n_change_points values."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n=10, n_change_points=n_change_points)


@pytest.mark.parametrize("cpts", [5, [3, 5, 7], np.arange(1, 10), None])
def test_generate_piecewise_normal_data_valid_cpts(cpts: list[int]):
    """Test that the function generates data with the correct change points."""
    n = 10
    p = 2
    df, params = generate_piecewise_normal_data(
        n=n, p=p, change_points=cpts, return_params=True
    )

    if cpts is not None:
        cpts = [cpts] if isinstance(cpts, int) else cpts
        cpts = np.asarray(cpts)
        assert np.all(params["change_points"] == cpts)
    else:
        assert len(params["change_points"]) >= 0


@pytest.mark.parametrize("cpts", [-1, [1, 1], [1, 14]])
def test_generate_piecewise_normal_data_invalid_cpts(cpts: list[int]):
    """Test that the function raises ValueError for invalid change points."""
    with pytest.raises(ValueError):
        generate_piecewise_normal_data(n=10, p=2, change_points=cpts)
