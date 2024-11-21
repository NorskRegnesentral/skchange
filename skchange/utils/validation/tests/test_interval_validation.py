import numpy as np
import pytest

from skchange.utils.validation.intervals import check_array_intervals


def test_check_array_intervals_valid():
    intervals = np.array([[1, 3], [4, 6], [7, 9]])
    result = check_array_intervals(intervals)
    assert np.array_equal(result, intervals)


def test_check_array_intervals_invalid_ndim():
    intervals = np.array([1, 2, 3])
    with pytest.raises(ValueError, match="The intervals must be a 2D array."):
        check_array_intervals(intervals)


def test_check_array_intervals_invalid_dtype():
    intervals = np.array([[1.0, 3.0], [4.0, 6.0], [7.0, 9.0]])
    with pytest.raises(ValueError, match="The intervals must be of integer type."):
        check_array_intervals(intervals)


def test_check_array_intervals_invalid_last_dim_size():
    intervals = np.array([[1, 3, 5], [4, 6, 8], [7, 9, 11]])
    with pytest.raises(
        ValueError,
        match=(
            "The intervals must be specified as an array with length 2 in the last"
            " dimension."
        ),
    ):
        check_array_intervals(intervals)


def test_check_array_intervals_not_strictly_increasing():
    intervals = np.array([[1, 3], [6, 4], [7, 9]])
    with pytest.raises(
        ValueError, match="All rows in the intervals must be strictly increasing."
    ):
        check_array_intervals(intervals)


def test_check_array_intervals_invalid_min_size():
    intervals = np.array([[1, 2], [4, 6], [7, 9]])
    with pytest.raises(ValueError, match="The interval sizes must be at least 3."):
        check_array_intervals(intervals, min_size=3)
