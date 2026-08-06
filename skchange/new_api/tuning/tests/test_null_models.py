"""Tests for ``skchange.new_api.tuning._null_models``."""

import copy

import numpy as np
import pytest

from skchange.new_api.tuning import (
    BaseNullSampler,
    GaussianSampler,
    PermutationSampler,
)
from skchange.new_api.tuning._null_models import _resolve_sampler

# --------------------------------------------------------------------------- #
# PermutationSampler
# --------------------------------------------------------------------------- #


def test_permutation_sampler_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    out = PermutationSampler().sample(X, 40, rng)
    assert out.shape == (40, 3)
    assert out.dtype == np.float64


def test_permutation_sampler_rows_come_from_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    out = PermutationSampler(replace=True).sample(X, 200, rng)
    original = {tuple(r) for r in X}
    assert all(tuple(r) in original for r in out)


def test_permutation_sampler_without_replacement_preserves_marginals():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    out = PermutationSampler(replace=False).sample(X, 60, rng)
    for j in range(2):
        np.testing.assert_array_equal(np.sort(out[:, j]), np.sort(X[:, j]))


def test_permutation_sampler_without_replacement_too_many_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="cannot draw"):
        PermutationSampler(replace=False).sample(X, 50, rng)


# --------------------------------------------------------------------------- #
# GaussianSampler
# --------------------------------------------------------------------------- #


def test_gaussian_sampler_shape_and_dtype():
    rng = np.random.default_rng(0)
    X = np.empty((0, 4))  # only shape[1] is used
    out = GaussianSampler().sample(X, 30, rng)
    assert out.shape == (30, 4)
    assert out.dtype == np.float64


def test_gaussian_sampler_respects_mean_and_std():
    rng = np.random.default_rng(0)
    X = np.empty((0, 1))
    out = GaussianSampler(mean=5.0, std=2.0).sample(X, 100_000, rng)
    assert abs(out.mean() - 5.0) < 0.05
    assert abs(out.std() - 2.0) < 0.05


def test_gaussian_sampler_ignores_reference_values():
    rng = np.random.default_rng(0)
    X_shifted = np.full((5, 2), 100.0)
    out = GaussianSampler().sample(X_shifted, 5_000, rng)
    # Centred at 0, unaffected by the huge values in X.
    assert abs(out.mean()) < 0.1


# --------------------------------------------------------------------------- #
# _resolve_sampler
# --------------------------------------------------------------------------- #


def test_resolve_sampler_string_aliases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    perm_fn = _resolve_sampler("permutation")
    gauss_fn = _resolve_sampler("gaussian")
    assert perm_fn(X, 5, rng).shape == (5, 2)
    assert gauss_fn(X, 5, rng).shape == (5, 2)


def test_resolve_sampler_instance_returns_bound_sample_method():
    inst = PermutationSampler(replace=True)
    fn = _resolve_sampler(inst)
    # The returned callable dispatches to the instance's `sample`.
    assert fn.__self__ is inst  # type: ignore[attr-defined]


def test_resolve_sampler_callable_passthrough():
    f = lambda X, n, rng: rng.normal(size=(n, X.shape[1]))  # noqa: E731
    assert _resolve_sampler(f) is f


def test_resolve_sampler_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown sampler"):
        _resolve_sampler("nope")


def test_resolve_sampler_bad_type_raises():
    with pytest.raises(TypeError):
        _resolve_sampler(123)


# --------------------------------------------------------------------------- #
# BaseNullSampler contract
# --------------------------------------------------------------------------- #


def test_base_null_sampler_sample_is_not_implemented():
    rng = np.random.default_rng(0)
    X = np.zeros((5, 2))
    with pytest.raises(NotImplementedError):
        BaseNullSampler().sample(X, 3, rng)


def test_samplers_repr_shows_params():
    assert repr(PermutationSampler(replace=True)) == "PermutationSampler(replace=True)"
    assert repr(GaussianSampler(mean=1.0, std=2.0)) == (
        "GaussianSampler(mean=1.0, std=2.0)"
    )


def test_samplers_are_deepcopyable():
    for sampler in [PermutationSampler(replace=True), GaussianSampler(std=2.0)]:
        cloned = copy.deepcopy(sampler)
        assert vars(cloned) == vars(sampler)
        assert cloned is not sampler
