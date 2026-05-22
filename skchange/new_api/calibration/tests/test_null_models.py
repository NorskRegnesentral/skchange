"""Contract tests for null models used in calibration."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.calibration._null_models import (
    BaseDataSampler,
    BaseParametricSampler,
    BlockBootstrapSampler,
    GaussianMCSampler,
    MCSimulator,
    PermutationSampler,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N, _P = 80, 5
_RNG = np.random.default_rng(0)
_X = _RNG.normal(size=(_N, _P))

_ALL_DATA_SAMPLERS = [
    PermutationSampler(),
    PermutationSampler(replace=False),
    PermutationSampler(replace=True),
    BlockBootstrapSampler(),
    BlockBootstrapSampler(block_length=10),
]

_ALL_PARAMETRIC_SAMPLERS = [
    GaussianMCSampler(),
    GaussianMCSampler(mean=0.0, std=1.0),
    MCSimulator(lambda n, p, rng: rng.normal(size=(n, p))),
]


@pytest.fixture(params=_ALL_DATA_SAMPLERS, ids=repr)
def data_sampler(request):
    return clone(request.param)


@pytest.fixture(params=_ALL_PARAMETRIC_SAMPLERS, ids=repr)
def parametric_sampler(request):
    return clone(request.param)


# ---------------------------------------------------------------------------
# Type hierarchy
# ---------------------------------------------------------------------------


def test_data_samplers_are_base_data_sampler(data_sampler):
    assert isinstance(data_sampler, BaseDataSampler)


def test_parametric_samplers_are_base_parametric_sampler(parametric_sampler):
    assert isinstance(parametric_sampler, BaseParametricSampler)


def test_data_samplers_not_parametric(data_sampler):
    assert not isinstance(data_sampler, BaseParametricSampler)


def test_parametric_samplers_not_data(parametric_sampler):
    assert not isinstance(parametric_sampler, BaseDataSampler)


# ---------------------------------------------------------------------------
# get_params / set_params / clone — both hierarchies
# ---------------------------------------------------------------------------


def test_data_sampler_get_set_params_roundtrip(data_sampler):
    params = data_sampler.get_params(deep=True)
    data_sampler.set_params(**params)
    assert data_sampler.get_params(deep=True) == params


def test_parametric_sampler_get_set_params_roundtrip(parametric_sampler):
    params = parametric_sampler.get_params(deep=True)
    parametric_sampler.set_params(**params)
    assert parametric_sampler.get_params(deep=True) == params


def test_data_sampler_clone(data_sampler):
    cloned = clone(data_sampler)
    assert type(cloned) is type(data_sampler)


def test_parametric_sampler_clone(parametric_sampler):
    cloned = clone(parametric_sampler)
    assert type(cloned) is type(parametric_sampler)


# ---------------------------------------------------------------------------
# BaseDataSampler: fit required before sample
# ---------------------------------------------------------------------------


def test_data_sampler_sample_before_fit_raises(data_sampler):
    """sample() before fit() must raise NotFittedError."""
    rng = np.random.default_rng(1)
    with pytest.raises(NotFittedError):
        data_sampler.sample(_N, rng)


def test_data_sampler_fit_returns_self(data_sampler):
    assert data_sampler.fit(_X) is data_sampler


def test_data_sampler_sample_shape(data_sampler):
    rng = np.random.default_rng(2)
    data_sampler.fit(_X)
    out = data_sampler.sample(_N, rng)
    assert out.shape == (_N, _P)


def test_data_sampler_sample_different_n(data_sampler):
    rng = np.random.default_rng(3)
    data_sampler.fit(_X)
    out = data_sampler.sample(30, rng)
    assert out.shape == (30, _P)


def test_data_sampler_sample_returns_float64(data_sampler):
    rng = np.random.default_rng(4)
    data_sampler.fit(_X)
    out = data_sampler.sample(_N, rng)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# BaseParametricSampler: no fit needed
# ---------------------------------------------------------------------------


def test_parametric_sampler_sample_without_fit(parametric_sampler):
    """sample(n, n_features, rng) must work without calling fit() first."""
    rng = np.random.default_rng(1)
    out = parametric_sampler.sample(_N, _P, rng)
    assert out.shape == (_N, _P)


def test_parametric_sampler_sample_returns_float64(parametric_sampler):
    rng = np.random.default_rng(2)
    out = parametric_sampler.sample(_N, _P, rng)
    assert out.dtype == np.float64


def test_parametric_sampler_sample_different_n_and_p(parametric_sampler):
    rng = np.random.default_rng(3)
    out = parametric_sampler.sample(30, 7, rng)
    assert out.shape == (30, 7)


# ---------------------------------------------------------------------------
# PermutationSampler specifics
# ---------------------------------------------------------------------------


def test_permutation_strict_exact_marginals():
    """replace=False: sorted output columns must equal sorted input columns."""
    model = PermutationSampler(replace=False).fit(_X)
    rng = np.random.default_rng(6)
    out = model.sample(_N, rng)
    for j in range(_P):
        np.testing.assert_array_almost_equal(np.sort(out[:, j]), np.sort(_X[:, j]))


def test_permutation_bootstrap_rows_come_from_X():
    """replace=True: every row of the sample must be a row of X."""
    model = PermutationSampler(replace=True).fit(_X)
    rng = np.random.default_rng(7)
    out = model.sample(_N, rng)
    for row in out:
        assert any(np.allclose(row, x_row) for x_row in _X), (
            f"Row {row} is not found in training data."
        )


def test_permutation_bootstrap_allows_repeats():
    """replace=True may produce repeated rows."""
    model = PermutationSampler(replace=True).fit(_X)
    rng = np.random.default_rng(8)
    out = model.sample(500, rng)
    _, counts = np.unique(out, axis=0, return_counts=True)
    assert np.any(counts > 1)


# ---------------------------------------------------------------------------
# BlockBootstrapSampler specifics
# ---------------------------------------------------------------------------


def test_block_bootstrap_blocks_are_contiguous():
    """Consecutive samples within a block must come from consecutive rows of X."""
    model = BlockBootstrapSampler(block_length=10).fit(_X)
    rng = np.random.default_rng(9)
    out = model.sample(_N, rng)
    assert out.shape == (_N, _P)
    block_length = 10
    for block_start in range(0, _N, block_length):
        block_end = min(block_start + block_length, _N)
        for i in range(block_start, block_end - 1):
            matches = np.where(np.all(np.isclose(_X, out[i]), axis=1))[0]
            if len(matches) == 0:
                continue
            idx = matches[0]
            next_idx = (idx + 1) % _N
            np.testing.assert_array_almost_equal(out[i + 1], _X[next_idx])


# ---------------------------------------------------------------------------
# GaussianMCSampler specifics
# ---------------------------------------------------------------------------


def test_gaussian_mc_sampler_default_params():
    """Default GaussianMCSampler uses mean=0, std=1."""
    model = GaussianMCSampler()
    assert model.mean == 0.0
    assert model.std == 1.0


def test_gaussian_mc_sampler_mean_close_to_param():
    """With many samples, output mean should be close to GaussianMCSampler.mean."""
    model = GaussianMCSampler(mean=2.5, std=1.0)
    rng = np.random.default_rng(10)
    out = model.sample(10_000, _P, rng)
    np.testing.assert_allclose(out.mean(axis=0), 2.5, atol=0.1)


def test_gaussian_mc_sampler_std_close_to_param():
    """With many samples, output std should be close to GaussianMCSampler.std."""
    model = GaussianMCSampler(mean=0.0, std=3.0)
    rng = np.random.default_rng(11)
    out = model.sample(10_000, _P, rng)
    np.testing.assert_allclose(out.std(axis=0), 3.0, atol=0.1)


# ---------------------------------------------------------------------------
# MCSimulator specifics
# ---------------------------------------------------------------------------


def test_mc_simulator_calls_dgp_with_correct_signature():
    """The DGP callable must be called with (n, n_features, rng)."""
    calls = []

    def dgp(n, p, rng):
        calls.append((n, p))
        return rng.normal(size=(n, p))

    model = MCSimulator(dgp)
    rng = np.random.default_rng(12)
    out = model.sample(30, _P, rng)
    assert len(calls) == 1
    assert calls[0] == (30, _P)
    assert out.shape == (30, _P)


def test_mc_simulator_dgp_return_shape():
    """MCSimulator must forward whatever shape the DGP returns."""
    model = MCSimulator(lambda n, p, rng: rng.normal(size=(n, p)))
    rng = np.random.default_rng(13)
    out = model.sample(20, _P, rng)
    assert out.shape == (20, _P)
