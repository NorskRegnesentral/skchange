"""Tests for ``skchange.tuning._null_models``."""

import copy

import numpy as np
import pytest

from skchange.tuning import (
    BaseNullSampler,
    BlockBootstrapSampler,
    GaussianSampler,
    PermutationSampler,
)
from skchange.tuning._null_models import (
    resolve_sampler,
    sampler_requires_data,
)

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
# BlockBootstrapSampler
# --------------------------------------------------------------------------- #


def _row_indices(out, X):
    """Map each row of ``out`` back to its row index in ``X`` (rows unique)."""
    lookup = {tuple(row): i for i, row in enumerate(X)}
    return [lookup[tuple(row)] for row in out]


def test_block_bootstrap_shape_and_rows_from_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    out = BlockBootstrapSampler(block_length=5).sample(X, 40, rng)
    assert out.shape == (40, 3)
    assert out.dtype == np.float64
    original = {tuple(r) for r in X}
    assert all(tuple(r) in original for r in out)


def test_block_bootstrap_wraps_and_warns_when_pool_too_small():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 2))
    with pytest.warns(UserWarning, match="reference"):
        out = BlockBootstrapSampler(block_length=3).sample(X, 20, rng)
    assert out.shape == (20, 2)


def test_block_bootstrap_default_block_length_is_cube_root():
    sampler = BlockBootstrapSampler()  # block_length=None
    assert sampler._effective_block_length(1000) == max(1, int(1000 ** (1 / 3)))
    assert sampler._effective_block_length(8) == 2
    assert sampler._effective_block_length(1) == 1


def test_block_bootstrap_length_one_is_iid():
    """block_length=1 draws each row independently (an i.i.d. row bootstrap)."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    out = BlockBootstrapSampler(block_length=1).sample(X, 500, rng)
    assert out.shape == (500, 2)
    # With independent single-row draws, adjacent output rows are contiguous in
    # X only by chance, so the fraction of contiguous adjacent pairs stays low.
    idx = _row_indices(out, X)
    contiguous = sum((idx[i] + 1) % len(X) == idx[i + 1] for i in range(len(idx) - 1))
    assert contiguous < 0.2 * len(idx)


def test_block_bootstrap_blocks_are_contiguous_slices():
    """For block_length>1, each block is a contiguous wrap-around slice of X."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 1))
    block_length = 6
    n_samples = 37
    out = BlockBootstrapSampler(block_length=block_length).sample(X, n_samples, rng)
    idx = _row_indices(out, X)
    n_ref = len(X)
    for start in range(0, n_samples, block_length):
        block = idx[start : start + block_length]
        for a, b in zip(block, block[1:]):
            assert (a + 1) % n_ref == b, "rows within a block must be contiguous"


def test_block_bootstrap_invalid_block_length_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    with pytest.raises(ValueError, match="block_length"):
        BlockBootstrapSampler(block_length=0).sample(X, 10, rng)


def test_block_bootstrap_requires_reference_data():
    assert BlockBootstrapSampler().requires_reference_data is True
    assert sampler_requires_data("block_bootstrap") is True


def test_block_bootstrap_reproducible_with_seed():
    X = np.random.default_rng(0).normal(size=(50, 2))
    a = BlockBootstrapSampler(block_length=4).sample(X, 30, np.random.default_rng(7))
    b = BlockBootstrapSampler(block_length=4).sample(X, 30, np.random.default_rng(7))
    c = BlockBootstrapSampler(block_length=4).sample(X, 30, np.random.default_rng(8))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_block_bootstrap_alias_resolves():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 2))
    fn = resolve_sampler("block_bootstrap")
    assert fn(X, 20, rng).shape == (20, 2)


# --------------------------------------------------------------------------- #
# resolve_sampler
# --------------------------------------------------------------------------- #


def test_resolve_sampler_string_aliases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 2))
    perm_fn = resolve_sampler("permutation")
    gauss_fn = resolve_sampler("gaussian")
    assert perm_fn(X, 5, rng).shape == (5, 2)
    assert gauss_fn(X, 5, rng).shape == (5, 2)


def test_resolve_sampler_instance_returns_bound_sample_method():
    inst = PermutationSampler(replace=True)
    fn = resolve_sampler(inst)
    # The returned callable dispatches to the instance's `sample`.
    assert fn.__self__ is inst  # type: ignore[attr-defined]


def test_resolve_sampler_callable_passthrough():
    f = lambda X, n, rng: rng.normal(size=(n, X.shape[1]))  # noqa: E731
    assert resolve_sampler(f) is f


def test_resolve_sampler_unknown_string_raises():
    with pytest.raises(ValueError, match="Unknown sampler"):
        resolve_sampler("nope")


def test_resolve_sampler_bad_type_raises():
    with pytest.raises(TypeError):
        resolve_sampler(123)


def test_sampler_requires_data_unknown_string_is_conservative():
    """An unrecognised string alias is treated as data-based (returns True).

    ``sampler_requires_data`` cannot introspect a name it does not know, so it
    errs on the safe side and reports that reference data is required. The
    unknown name is only rejected later, when ``resolve_sampler`` runs.
    """
    assert sampler_requires_data("definitely-not-a-real-sampler") is True


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
