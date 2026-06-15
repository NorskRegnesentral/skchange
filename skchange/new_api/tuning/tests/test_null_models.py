"""Tests for ``skchange.new_api.tuning._null_models``."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.tuning import GaussianMCSampler, PermutationSampler
from skchange.new_api.tuning._null_models import _resolve_sampler, make_null_draw


def test_permutation_sampler_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    sampler = PermutationSampler().fit(X)
    out = sampler.sample(40, rng)
    assert out.shape == (40, 3)
    assert out.dtype == np.float64


def test_permutation_sampler_rows_come_from_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    sampler = PermutationSampler(replace=True).fit(X)
    out = sampler.sample(200, rng)
    # Every sampled row must be one of the original rows.
    original = {tuple(r) for r in X}
    assert all(tuple(r) in original for r in out)


def test_permutation_sampler_without_replacement_preserves_marginals():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    sampler = PermutationSampler(replace=False).fit(X)
    out = sampler.sample(60, rng)  # full permutation
    for j in range(2):
        np.testing.assert_array_equal(np.sort(out[:, j]), np.sort(X[:, j]))


def test_permutation_sampler_without_replacement_too_many_raises():
    rng = np.random.default_rng(0)
    sampler = PermutationSampler(replace=False).fit(rng.normal(size=(20, 2)))
    with pytest.raises(ValueError, match="cannot draw"):
        sampler.sample(50, rng)


def test_permutation_sampler_sample_before_fit_raises():
    with pytest.raises(NotFittedError):
        PermutationSampler().sample(10, np.random.default_rng(0))


def test_gaussian_sampler_shape_dtype_and_no_fit_needed():
    rng = np.random.default_rng(0)
    out = GaussianMCSampler().sample(30, 4, rng)  # no fit() call
    assert out.shape == (30, 4)
    assert out.dtype == np.float64


def test_gaussian_sampler_respects_mean_and_std():
    rng = np.random.default_rng(0)
    out = GaussianMCSampler(mean=5.0, std=2.0).sample(100000, 1, rng)
    assert abs(out.mean() - 5.0) < 0.05
    assert abs(out.std() - 2.0) < 0.05


def test_resolve_sampler_aliases_and_passthrough():
    assert isinstance(_resolve_sampler("permutation"), PermutationSampler)
    assert isinstance(_resolve_sampler("gaussian"), GaussianMCSampler)
    inst = PermutationSampler()
    assert _resolve_sampler(inst) is inst
    f = lambda n, p, rng: rng.normal(size=(n, p))  # noqa: E731
    assert _resolve_sampler(f) is f


def test_resolve_sampler_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown sampler"):
        _resolve_sampler("nope")


def test_resolve_sampler_bad_type_raises():
    with pytest.raises(TypeError):
        _resolve_sampler(123)


def test_make_null_draw_data_based_uses_calib_then_x():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    X_calib = rng.normal(size=(120, 2)) + 100.0  # clearly distinct
    draw = make_null_draw("permutation", X, X_calib, n_samples=50, n_features=2)
    out = draw(rng)
    assert out.shape == (50, 2)
    assert out.mean() > 50  # drew from X_calib, not X


def test_make_null_draw_parametric_ignores_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    draw = make_null_draw("gaussian", X, None, n_samples=50, n_features=2)
    assert draw(rng).shape == (50, 2)


def test_make_null_draw_callable_escape_hatch():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    draw = make_null_draw(
        lambda n, p, r: r.standard_t(df=5, size=(n, p)), X, None, 40, 2
    )
    assert draw(rng).shape == (40, 2)


def test_make_null_draw_feature_mismatch_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    X_calib = rng.normal(size=(100, 3))
    with pytest.raises(ValueError, match="features"):
        make_null_draw("permutation", X, X_calib, n_samples=50, n_features=2)


def test_samplers_are_sklearn_cloneable():
    for sampler in [PermutationSampler(replace=True), GaussianMCSampler(std=2.0)]:
        cloned = clone(sampler)
        assert cloned.get_params() == sampler.get_params()
