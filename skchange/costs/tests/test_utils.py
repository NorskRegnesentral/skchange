import numpy as np
import pytest

from skchange.costs.utils import check_univariate_scale


def test_check_univariate_scale_with_scalar():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scale = 2.0
    result = check_univariate_scale(scale, X)
    assert np.array_equal(result, np.array([2.0]))


def test_check_univariate_scale_with_array():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scale = np.array([1.0, 2.0, 3.0])
    result = check_univariate_scale(scale, X)
    assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


def test_check_univariate_scale_with_wrong_length():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scale = np.array([1.0, 2.0])
    with pytest.raises(ValueError, match="mean must have length 1 or X.shape\\[1\\]"):
        check_univariate_scale(scale, X)


def test_check_univariate_scale_with_non_positive_values():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scale = np.array([1.0, 0.0, 3.0])
    with pytest.raises(ValueError, match="scales must be positive."):
        check_univariate_scale(scale, X)


def test_check_univariate_scale_with_negative_values():
    X = np.array([[1, 2, 3], [4, 5, 6]])
    scale = np.array([1.0, -2.0, 3.0])
    with pytest.raises(ValueError, match="scales must be positive."):
        check_univariate_scale(scale, X)
