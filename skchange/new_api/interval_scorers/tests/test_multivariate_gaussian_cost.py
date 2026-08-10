"""Tests for MultivariateGaussianCost cumulative moment caching."""

import numpy as np
import pytest

from skchange.new_api.interval_scorers import (
    MultivariateGaussianCost,
    MultivariateGaussianSaving,
    MultivariateGaussianScore,
)
from skchange.new_api.interval_scorers._costs.multivariate_gaussian_cost import (
    MAX_COV_CACHE_ELEMENTS,
)


def test_precompute_cumulative_moments():
    """Cached moments have a zero prefix and contain the expected sums."""
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0]])
    cost = MultivariateGaussianCost(store_cov=True).fit(X)

    cache = cost.precompute(X)

    assert cache["store_cov"] is True
    assert set(cache) == {"feature_sums", "outer_product_sums", "store_cov"}
    assert cache["feature_sums"].shape == (len(X) + 1, X.shape[1])
    assert cache["outer_product_sums"].shape == (
        len(X) + 1,
        X.shape[1],
        X.shape[1],
    )
    np.testing.assert_array_equal(cache["feature_sums"][0], 0.0)
    np.testing.assert_array_equal(cache["outer_product_sums"][0], 0.0)
    np.testing.assert_allclose(cache["feature_sums"][1:], np.cumsum(X, axis=0))
    outer_products = np.einsum("ni,nj->nij", X, X)
    np.testing.assert_allclose(
        cache["outer_product_sums"][1:], np.cumsum(outer_products, axis=0)
    )


@pytest.mark.parametrize(
    "shape, expected_store_cov",
    [
        ((100, 20), True),
        ((10_001, 100), False),
        ((250, 10), True),
        ((2000, 500), False),
    ],
)
def test_automatic_cache_selection(shape, expected_store_cov):
    """Automatic caching uses the inclusive n_samples * n_features**2 threshold."""
    X = np.zeros(shape)
    cost = MultivariateGaussianCost().fit(X)

    cache = cost.precompute(X)

    assert cost.store_cov is None
    assert cache["store_cov"] is expected_store_cov
    assert cache["store_cov"] is (shape[0] * shape[1] ** 2 <= MAX_COV_CACHE_ELEMENTS)


@pytest.mark.parametrize(
    "store_cov, shape",
    [
        (True, (10_001, 1)),
        (True, (2, 101)),
        (False, (10, 2)),
    ],
)
def test_explicit_cache_selection_overrides_size(store_cov, shape):
    """Explicit cache settings override automatic size selection."""
    X = np.zeros(shape)
    cost = MultivariateGaussianCost(store_cov=store_cov).fit(X)

    cache = cost.precompute(X)

    assert cache["store_cov"] is store_cov


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
    cached_cost = MultivariateGaussianCost(store_cov=True).fit(X)
    uncached_cost = MultivariateGaussianCost(store_cov=False).fit(X)

    actual = cached_cost.evaluate(cached_cost.precompute(X), interval_specs)
    expected = uncached_cost.evaluate(uncached_cost.precompute(X), interval_specs)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_cached_and_uncached_invalid_interval_costs_are_equal():
    """Both strategies return infinity for undersized and singular intervals."""
    X = np.ones((10, 3))
    interval_specs = np.array([[0, 2], [0, 10]])
    cached_cost = MultivariateGaussianCost(store_cov=True).fit(X)
    uncached_cost = MultivariateGaussianCost(store_cov=False).fit(X)

    actual = cached_cost.evaluate(cached_cost.precompute(X), interval_specs)
    expected = uncached_cost.evaluate(uncached_cost.precompute(X), interval_specs)

    np.testing.assert_array_equal(actual, expected)
    assert np.all(np.isinf(actual))


@pytest.mark.parametrize(
    "scorer_class, scorer_kwargs, interval_specs",
    [
        (
            MultivariateGaussianScore,
            {"apply_bartlett_correction": False},
            np.array([[0, 30, 60], [10, 55, 100]]),
        ),
        (
            MultivariateGaussianSaving,
            {"baseline_mean": 0.0, "baseline_cov": 1.0},
            np.array([[0, 60], [10, 100]]),
        ),
    ],
)
def test_cached_and_uncached_multivariate_gaussian_variants_are_equal(
    scorer_class, scorer_kwargs, interval_specs
):
    """Score and saving variants share the equivalent moment cache path."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 3))
    cached_scorer = scorer_class(store_cov=True, **scorer_kwargs).fit(X)
    uncached_scorer = scorer_class(store_cov=False, **scorer_kwargs).fit(X)

    actual = cached_scorer.evaluate(cached_scorer.precompute(X), interval_specs)
    expected = uncached_scorer.evaluate(uncached_scorer.precompute(X), interval_specs)

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
