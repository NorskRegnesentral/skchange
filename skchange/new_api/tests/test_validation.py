"""Unit tests for skchange.new_api.utils.validation."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from skchange.new_api.utils.validation import (
    check_interval_specs,
    check_time_col,
    validate_data,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyEstimator(BaseEstimator):
    """Minimal estimator used to test validate_data side effects."""


# ---------------------------------------------------------------------------
# validate_data — n_samples_in_ side effect
# ---------------------------------------------------------------------------


def test_validate_data_sets_n_samples_in_on_fit():
    """validate_data with reset=True sets n_samples_in_ on the estimator."""
    est = _DummyEstimator()
    X = np.ones((10, 2))
    validate_data(est, X, reset=True)
    assert est.n_samples_in_ == 10


def test_validate_data_does_not_overwrite_n_samples_in_on_transform():
    """validate_data with reset=False does not overwrite n_samples_in_."""
    est = _DummyEstimator()
    X_fit = np.ones((10, 2))
    validate_data(est, X_fit, reset=True)
    X_transform = np.ones((5, 2))
    validate_data(est, X_transform, reset=False)
    assert est.n_samples_in_ == 10  # unchanged


# ---------------------------------------------------------------------------
# check_time_col — happy path
# ---------------------------------------------------------------------------


def test_check_time_col_valid_does_not_raise():
    """Valid time column passes without error."""
    X = np.column_stack([np.arange(5, dtype=float), np.ones(5)])
    check_time_col(X, time_col=0, caller_name="Test")


# ---------------------------------------------------------------------------
# check_time_col — error paths
# ---------------------------------------------------------------------------


def test_check_time_col_out_of_range_raises():
    """time_col outside [0, n_features) raises ValueError."""
    X = np.column_stack([np.arange(5, dtype=float), np.ones(5)])
    for bad_col in (5, -1):
        with pytest.raises(ValueError, match="out of range"):
            check_time_col(X, time_col=bad_col, caller_name="Test")


def test_check_time_col_only_one_column_raises():
    """n_features < 2 raises ValueError (no value columns left)."""
    X = np.arange(5, dtype=float).reshape(-1, 1)
    with pytest.raises(ValueError, match="at least 2"):
        check_time_col(X, time_col=0, caller_name="Test")


def test_check_time_col_non_finite_raises():
    """Non-finite timestamps raise ValueError."""
    X = np.column_stack([np.array([0.0, np.nan, 2.0, 3.0, 4.0]), np.ones(5)])
    with pytest.raises(ValueError, match="non-finite"):
        check_time_col(X, time_col=0, caller_name="Test")


def test_check_time_col_monotonicity_raises():
    """Non-monotone and duplicate timestamps both raise ValueError."""
    values = np.ones(5)
    for timestamps in (
        np.array([0.0, 2.0, 1.0, 3.0, 4.0]),  # reversed pair
        np.array([0.0, 1.0, 1.0, 2.0, 3.0]),  # duplicate
    ):
        X = np.column_stack([timestamps, values])
        with pytest.raises(ValueError, match="strictly monotonically"):
            check_time_col(X, time_col=0, caller_name="Test")


# ---------------------------------------------------------------------------
# check_interval_specs — empty input
# ---------------------------------------------------------------------------


def test_check_interval_specs_empty_returns_empty():
    """Empty input is returned without error."""
    empty = np.empty((0, 2), dtype=np.intp)
    result = check_interval_specs(empty, n_cols=2)
    assert result.size == 0


# ---------------------------------------------------------------------------
# check_interval_specs — column count
# ---------------------------------------------------------------------------


def test_check_interval_specs_wrong_n_cols_raises():
    """Wrong number of columns raises ValueError."""
    specs = np.array([[0, 5, 10]])
    with pytest.raises(ValueError, match="must have 2 columns"):
        check_interval_specs(specs, n_cols=2, caller_name="Test")


def test_check_interval_specs_correct_n_cols_passes():
    """Correct number of columns passes."""
    specs = np.array([[0, 10], [5, 15]])
    result = check_interval_specs(specs, n_cols=2)
    assert result.shape == (2, 2)


# ---------------------------------------------------------------------------
# check_interval_specs — check_sorted
# ---------------------------------------------------------------------------


def test_check_interval_specs_sorted_passes():
    """Strictly increasing rows pass when check_sorted=True."""
    specs = np.array([[0, 5], [10, 20]])
    check_interval_specs(specs, n_cols=2, check_sorted=True)


def test_check_interval_specs_not_strictly_increasing_raises():
    """Reversed and equal-entry rows both raise ValueError when check_sorted=True."""
    for specs in (np.array([[5, 0]]), np.array([[5, 5]])):
        with pytest.raises(ValueError, match="strictly increasing"):
            check_interval_specs(specs, n_cols=2, check_sorted=True)


# ---------------------------------------------------------------------------
# check_interval_specs — min_size
# ---------------------------------------------------------------------------


def test_check_interval_specs_min_size_violated_raises():
    """Interval smaller than min_size raises ValueError."""
    specs = np.array([[0, 3]])  # width 3, min_size=5
    with pytest.raises(ValueError, match="differ by at least"):
        check_interval_specs(specs, n_cols=2, min_size=5)


def test_check_interval_specs_min_size_exactly_satisfied_passes():
    """Interval exactly equal to min_size passes."""
    specs = np.array([[0, 5]])
    check_interval_specs(specs, n_cols=2, min_size=5)


# ---------------------------------------------------------------------------
# check_interval_specs — n_samples bounds
# ---------------------------------------------------------------------------


def test_check_interval_specs_in_range_passes():
    """Entries within [0, n_samples] pass."""
    specs = np.array([[0, 10], [5, 20]])
    check_interval_specs(specs, n_cols=2, n_samples=20)


def test_check_interval_specs_out_of_bounds_raises():
    """Entries above n_samples and negative entries both raise ValueError."""
    for specs in (np.array([[0, 25]]), np.array([[-1, 10]])):
        with pytest.raises(ValueError, match="must be in"):
            check_interval_specs(specs, n_cols=2, n_samples=20)
