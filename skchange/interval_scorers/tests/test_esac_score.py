"""Tests for ESACScore.penalty_scale."""

import numpy as np
import pytest
from sklearn.base import clone

from skchange.interval_scorers import ESACScore


def _make_X(n=80, p=5, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


def test_default_penalty_scale_is_one():
    assert ESACScore().penalty_scale == 1.0


def test_penalty_scale_in_get_params():
    scorer = ESACScore(penalty_scale=2.5)
    assert scorer.get_params()["penalty_scale"] == 2.5


def test_penalty_scale_set_params_roundtrip():
    scorer = ESACScore()
    scorer.set_params(penalty_scale=3.0)
    assert scorer.penalty_scale == 3.0
    assert scorer.get_params()["penalty_scale"] == 3.0


def test_penalty_scale_clone():
    scorer = ESACScore(penalty_scale=1.8)
    cloned = clone(scorer)
    assert cloned.get_params()["penalty_scale"] == 1.8


def test_penalty_scale_scales_all_sparsity_penalties():
    """ESACScore(penalty_scale=c) must scale every sparsity_penalty_ by c."""
    X = _make_X()
    base = ESACScore(penalty_scale=1.0).fit(X)
    scaled = ESACScore(penalty_scale=3.0).fit(X)
    np.testing.assert_allclose(
        scaled.sparsity_penalties_, 3.0 * base.sparsity_penalties_
    )


def test_penalty_scale_preserves_dense_sparse_ratio():
    """penalty_scale must multiply both dense and sparse constants uniformly."""
    X = _make_X()
    base = ESACScore(penalty_scale=1.0).fit(X)
    scaled = ESACScore(penalty_scale=2.0).fit(X)
    # Element-wise ratio should all be 2.0
    ratios = scaled.sparsity_penalties_ / base.sparsity_penalties_
    np.testing.assert_allclose(ratios, 2.0)


def test_default_scale_matches_legacy_behaviour():
    """penalty_scale=1 must reproduce the same sparsity penalties as before."""
    X = _make_X()
    old = ESACScore(penalty_scale_dense=2.0, penalty_scale_sparse=1.5).fit(X)
    new = ESACScore(
        penalty_scale=1.0, penalty_scale_dense=2.0, penalty_scale_sparse=1.5
    ).fit(X)
    np.testing.assert_array_equal(old.sparsity_penalties_, new.sparsity_penalties_)


@pytest.mark.parametrize("c", [0.5, 1.0, 2.0, 5.0])
def test_penalty_scale_positive_values(c):
    X = _make_X()
    scorer = ESACScore(penalty_scale=c).fit(X)
    assert scorer.sparsity_penalties_.min() > 0.0
