"""Tests for MultivariateGaussianCost cumulative moment caching."""

import numpy as np
import pytest

from skchange.new_api.interval_scorers import MultivariateGaussianCost


def test_precompute_cumulative_moments():
    """Cached moments have a zero prefix and contain the expected sums."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
    cost = MultivariateGaussianCost(use_cache=True).fit(X)

    cache = cost.precompute(X)

    assert cache["use_cache"] is True
    assert set(cache) == {"sums", "outer_product_sums", "use_cache"}
    assert cache["sums"].shape == (len(X) + 1, X.shape[1])
    assert cache["outer_product_sums"].shape == (
        len(X) + 1,
        X.shape[1],
        X.shape[1],
    )
    np.testing.assert_array_equal(cache["sums"][0], 0.0)
    np.testing.assert_array_equal(cache["outer_product_sums"][0], 0.0)
    np.testing.assert_allclose(cache["sums"][1:], np.cumsum(X, axis=0))
    outer_products = np.einsum("ni,nj->nij", X, X)
    np.testing.assert_allclose(
        cache["outer_product_sums"][1:], np.cumsum(outer_products, axis=0)
    )


@pytest.mark.parametrize(
    "shape, expected_use_cache",
    [
        ((10_000, 1), True),
        ((10_001, 1), False),
        ((2, 100), True),
        ((2, 101), False),
    ],
)
def test_automatic_cache_selection(shape, expected_use_cache):
    """Automatic caching uses inclusive sample and feature thresholds."""
    X = np.zeros(shape)
    cost = MultivariateGaussianCost().fit(X)

    cache = cost.precompute(X)

    assert cost.use_cache is None
    assert cache["use_cache"] is expected_use_cache


@pytest.mark.parametrize(
    "use_cache, shape",
    [
        (True, (10_001, 1)),
        (True, (2, 101)),
        (False, (10, 2)),
    ],
)
def test_explicit_cache_selection_overrides_size(use_cache, shape):
    """Explicit cache settings override automatic size selection."""
    X = np.zeros(shape)
    cost = MultivariateGaussianCost(use_cache=use_cache).fit(X)

    cache = cost.precompute(X)

    assert cache["use_cache"] is use_cache


@pytest.mark.parametrize("n_features", [1, 3, 8])
def test_cached_and_uncached_costs_are_equal(n_features):
    """Cached moments reproduce direct segment covariance costs."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, n_features))
    interval_specs = np.array(
        [
            [0, 30],
            [7, 45],
            [50, 120],
            [0, 120],
        ]
    )
    cached_cost = MultivariateGaussianCost(use_cache=True).fit(X)
    uncached_cost = MultivariateGaussianCost(use_cache=False).fit(X)

    actual = cached_cost.evaluate(cached_cost.precompute(X), interval_specs)
    expected = uncached_cost.evaluate(uncached_cost.precompute(X), interval_specs)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_cached_and_uncached_invalid_interval_costs_are_equal():
    """Both strategies return infinity for undersized and singular intervals."""
    X = np.ones((10, 3))
    interval_specs = np.array([[0, 2], [0, 10]])
    cached_cost = MultivariateGaussianCost(use_cache=True).fit(X)
    uncached_cost = MultivariateGaussianCost(use_cache=False).fit(X)

    actual = cached_cost.evaluate(cached_cost.precompute(X), interval_specs)
    expected = uncached_cost.evaluate(uncached_cost.precompute(X), interval_specs)

    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isinf(actual))
