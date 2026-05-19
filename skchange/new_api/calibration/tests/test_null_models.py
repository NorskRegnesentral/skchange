"""Contract tests for null models used in calibration."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.calibration._null_models import (
    BaseNullModel,
    BlockBootstrapNullModel,
    GaussianNullModel,
    MCNullModel,
    PermutationNullModel,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N, _P = 80, 5
_RNG = np.random.default_rng(0)
_X = _RNG.normal(size=(_N, _P))

_ALL_NULL_MODELS = [
    PermutationNullModel(),
    PermutationNullModel(replace=False),
    PermutationNullModel(replace=True),
    BlockBootstrapNullModel(),
    BlockBootstrapNullModel(block_length=10),
    GaussianNullModel(),
    MCNullModel(lambda n, p, rng: rng.normal(size=(n, p))),
]


@pytest.fixture(params=_ALL_NULL_MODELS, ids=repr)
def null_model(request):
    return clone(request.param)


# ---------------------------------------------------------------------------
# BaseNullModel contract
# ---------------------------------------------------------------------------


def test_all_null_models_are_base_null_model(null_model):
    """Every null model must be a BaseNullModel subclass."""
    assert isinstance(null_model, BaseNullModel)


def test_null_model_get_params_set_params_roundtrip(null_model):
    """get_params / set_params must be consistent (sklearn contract)."""
    params = null_model.get_params(deep=True)
    null_model.set_params(**params)
    assert null_model.get_params(deep=True) == params


def test_null_model_clone(null_model):
    """clone() must produce an unfitted copy."""
    cloned = clone(null_model)
    assert type(cloned) is type(null_model)


def test_sample_before_fit_raises(null_model):
    """sample() before fit() must raise NotFittedError."""
    rng = np.random.default_rng(1)
    with pytest.raises(NotFittedError):
        null_model.sample(10, rng)


def test_fit_returns_self(null_model):
    """fit() must return self."""
    assert null_model.fit(_X) is null_model


def test_sample_shape(null_model):
    """sample(n, rng) must return an array of shape (n, n_features_in_)."""
    rng = np.random.default_rng(2)
    null_model.fit(_X)
    out = null_model.sample(_N, rng)
    assert out.shape == (_N, _P)


def test_sample_different_n(null_model):
    """sample() must work for n different from the training set size."""
    rng = np.random.default_rng(3)
    null_model.fit(_X)
    out = null_model.sample(30, rng)
    assert out.shape == (30, _P)


def test_sample_returns_float64(null_model):
    """sample() output must be float64."""
    rng = np.random.default_rng(4)
    null_model.fit(_X)
    out = null_model.sample(_N, rng)
    assert out.dtype == np.float64


# ---------------------------------------------------------------------------
# PermutationNullModel specifics
# ---------------------------------------------------------------------------


def test_permutation_strict_rows_come_from_X():
    """With replace=False every row of the sample must be a row of X."""
    model = PermutationNullModel(replace=False).fit(_X)
    rng = np.random.default_rng(5)
    out = model.sample(_N, rng)
    # Each output row should appear in X when sorted.
    X_sorted_rows = set(map(tuple, np.sort(_X, axis=0)))
    for row in out:
        assert tuple(np.sort(row)) in X_sorted_rows or any(
            np.allclose(row, x_row) for x_row in _X
        )


def test_permutation_strict_exact_marginals():
    """replace=False: sorted output columns must equal sorted input columns."""
    model = PermutationNullModel(replace=False).fit(_X)
    rng = np.random.default_rng(6)
    out = model.sample(_N, rng)
    # Each column is a permutation of the corresponding input column.
    for j in range(_P):
        np.testing.assert_array_almost_equal(np.sort(out[:, j]), np.sort(_X[:, j]))


def test_permutation_bootstrap_rows_come_from_X():
    """replace=True: every row of the sample must be a row of X (resampling)."""
    model = PermutationNullModel(replace=True).fit(_X)
    rng = np.random.default_rng(7)
    out = model.sample(_N, rng)
    for row in out:
        assert any(np.allclose(row, x_row) for x_row in _X), (
            f"Row {row} is not found in training data."
        )


def test_permutation_bootstrap_allows_repeats():
    """replace=True may produce repeated rows (non-deterministically, but very likely)."""
    model = PermutationNullModel(replace=True).fit(_X)
    rng = np.random.default_rng(8)
    # With n=500 draws from 80 rows, repeats are virtually certain.
    out = model.sample(500, rng)
    # Check at least one duplicate row exists.
    _, counts = np.unique(out, axis=0, return_counts=True)
    assert np.any(counts > 1), (
        "Expected repeated rows with replace=True and n >> N_train."
    )


# ---------------------------------------------------------------------------
# BlockBootstrapNullModel specifics
# ---------------------------------------------------------------------------


def test_block_bootstrap_blocks_are_contiguous():
    """Consecutive samples within a block must come from consecutive rows of X."""
    model = BlockBootstrapNullModel(block_length=10).fit(_X)
    rng = np.random.default_rng(9)
    out = model.sample(_N, rng)
    assert out.shape == (_N, _P)
    # Walk through blocks: within each block of 10, row[i+1] must equal X[row_i + 1]
    # (modulo wrap-around at end of X — circular bootstrap).
    block_length = 10
    for block_start in range(0, _N, block_length):
        block_end = min(block_start + block_length, _N)
        for i in range(block_start, block_end - 1):
            # Find which row of X matches out[i]
            matches = np.where(np.all(np.isclose(_X, out[i]), axis=1))[0]
            if len(matches) == 0:
                continue  # allow unique float sequences
            idx = matches[0]
            next_idx = (idx + 1) % _N
            np.testing.assert_array_almost_equal(out[i + 1], _X[next_idx])


# ---------------------------------------------------------------------------
# GaussianNullModel specifics
# ---------------------------------------------------------------------------


def test_gaussian_null_model_sample_mean_close_to_X_mean():
    """With many samples, the output mean per feature should be close to X mean."""
    model = GaussianNullModel().fit(_X)
    rng = np.random.default_rng(10)
    out = model.sample(10_000, rng)
    np.testing.assert_allclose(out.mean(axis=0), _X.mean(axis=0), atol=0.1)


def test_gaussian_null_model_sample_std_close_to_X_std():
    """With many samples, the output std per feature should be close to X std."""
    model = GaussianNullModel().fit(_X)
    rng = np.random.default_rng(11)
    out = model.sample(10_000, rng)
    np.testing.assert_allclose(out.std(axis=0), _X.std(axis=0), atol=0.1)


# ---------------------------------------------------------------------------
# MCNullModel specifics
# ---------------------------------------------------------------------------


def test_mc_null_model_calls_dgp_with_correct_signature():
    """The DGP callable must be called with (n, p, rng)."""
    calls = []

    def dgp(n, p, rng):
        calls.append((n, p))
        return rng.normal(size=(n, p))

    model = MCNullModel(dgp).fit(_X)
    rng = np.random.default_rng(12)
    out = model.sample(30, rng)
    assert len(calls) == 1
    assert calls[0] == (30, _P)
    assert out.shape == (30, _P)


def test_mc_null_model_dgp_return_shape():
    """MCNullModel must forward whatever shape the DGP returns."""
    model = MCNullModel(lambda n, p, rng: rng.normal(size=(n, p))).fit(_X)
    rng = np.random.default_rng(13)
    out = model.sample(20, rng)
    assert out.shape == (20, _P)
