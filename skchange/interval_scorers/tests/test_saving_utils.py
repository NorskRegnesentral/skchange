"""Unit tests for skchange.new_api.interval_scorers._savings._utils."""

import numpy as np
import pytest

from skchange.new_api.interval_scorers._savings._utils import (
    resolve_baseline_location,
    resolve_baseline_location_and_scatter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
X_2D = RNG.standard_normal((50, 3))  # 50 samples, 3 features


# ---------------------------------------------------------------------------
# resolve_baseline_location
# ---------------------------------------------------------------------------


def test_resolve_baseline_location_none_returns_median():
    """When param_value is None, the column-wise median of X is returned."""
    result = resolve_baseline_location(None, X_2D)
    np.testing.assert_array_equal(result, np.median(X_2D, axis=0))


def test_resolve_baseline_location_scalar_broadcasts():
    """A scalar is broadcast to an array of shape (n_features,)."""
    result = resolve_baseline_location(5.0, X_2D)
    np.testing.assert_array_equal(result, np.full(3, 5.0))


def test_resolve_baseline_location_array_passthrough():
    """A correct (n_features,) array is returned unchanged."""
    param = np.array([1.0, 2.0, 3.0])
    result = resolve_baseline_location(param, X_2D)
    np.testing.assert_array_equal(result, param)


def test_resolve_baseline_location_wrong_shape_raises():
    """An array with wrong shape raises ValueError."""
    with pytest.raises(ValueError, match="must be a scalar or array of shape"):
        resolve_baseline_location(np.array([1.0, 2.0]), X_2D)  # shape (2,), need (3,)


# ---------------------------------------------------------------------------
# resolve_baseline_location_and_scatter — both-None branches
# ---------------------------------------------------------------------------


def test_resolve_both_none_uses_mcd():
    """When both params are None, MCD estimates are returned."""
    mean, scatter = resolve_baseline_location_and_scatter(None, None, X_2D)
    assert mean.shape == (3,)
    assert scatter.shape == (3, 3)
    # Scatter must be SPD: positive log-determinant
    sign, _ = np.linalg.slogdet(scatter)
    assert sign > 0


def test_resolve_both_none_n_leq_p_raises():
    """Both None with n <= p must raise ValueError."""
    X_thin = RNG.standard_normal((2, 3))  # n=2, p=3
    with pytest.raises(ValueError, match="n_samples"):
        resolve_baseline_location_and_scatter(None, None, X_thin)


# ---------------------------------------------------------------------------
# resolve_baseline_location_and_scatter — scatter estimated from data
# ---------------------------------------------------------------------------


def test_resolve_mean_given_scatter_none_estimates_scatter():
    """With scatter=None, scatter is estimated as biased sample scatter matrix."""
    mean_param = np.zeros(3)
    mean, scatter = resolve_baseline_location_and_scatter(mean_param, None, X_2D)
    np.testing.assert_array_equal(mean, mean_param)
    assert scatter.shape == (3, 3)
    # Must agree with manual computation
    centered = X_2D - mean_param
    expected = (centered.T @ centered) / len(X_2D)
    np.testing.assert_allclose(scatter, expected)


def test_resolve_mean_none_scatter_none_estimates_both():
    """With mean=None and scatter=None (implicitly via both-None check bypassed)
    is handled by the MCD branch, tested separately. Here we ensure that when
    mean is None but scatter is supplied, mean falls back to median."""
    scatter_param = np.eye(3)
    mean, scatter = resolve_baseline_location_and_scatter(None, scatter_param, X_2D)
    np.testing.assert_array_equal(mean, np.median(X_2D, axis=0))
    np.testing.assert_array_equal(scatter, scatter_param)


def test_resolve_scatter_none_n_leq_p_raises():
    """Scatter=None with n <= p raises ValueError even when mean is provided."""
    X_thin = RNG.standard_normal((2, 3))  # n=2, p=3
    with pytest.raises(ValueError, match="n_samples"):
        resolve_baseline_location_and_scatter(np.zeros(3), None, X_thin)


# ---------------------------------------------------------------------------
# resolve_baseline_location_and_scatter — scatter given explicitly
# ---------------------------------------------------------------------------


def test_resolve_scatter_scalar_broadcasts_to_identity():
    """A positive scalar scatter is broadcast to scalar * I."""
    mean, scatter = resolve_baseline_location_and_scatter(np.zeros(3), 2.0, X_2D)
    np.testing.assert_array_equal(scatter, 2.0 * np.eye(3))


def test_resolve_scatter_non_positive_scalar_raises():
    """A non-positive (negative or zero) scalar scatter raises ValueError."""
    for bad_value in (-1.0, 0.0):
        with pytest.raises(ValueError, match="strictly positive"):
            resolve_baseline_location_and_scatter(np.zeros(3), bad_value, X_2D)


def test_resolve_scatter_matrix_passthrough():
    """A correct (p, p) scatter matrix is returned unchanged."""
    scatter_param = np.eye(3) * 3.0
    _, scatter = resolve_baseline_location_and_scatter(np.zeros(3), scatter_param, X_2D)
    np.testing.assert_array_equal(scatter, scatter_param)


def test_resolve_scatter_wrong_shape_raises():
    """A scatter matrix with wrong shape raises ValueError."""
    bad_scatter = np.eye(2)  # (2, 2) instead of (3, 3)
    with pytest.raises(ValueError, match="must be a scalar or array of shape"):
        resolve_baseline_location_and_scatter(np.zeros(3), bad_scatter, X_2D)
