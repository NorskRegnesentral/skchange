"""Tests for PELT.penalty_scale and PELT._unpenalised_change_scores."""

import numpy as np
import pytest
from sklearn.base import clone

from skchange.new_api.detectors import PELT


def _make_X(n=80, p=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


# --------------------------------------------------------------------------- #
# penalty_scale parameter
# --------------------------------------------------------------------------- #


def test_pelt_default_penalty_scale_is_one():
    assert PELT().penalty_scale == 1.0


def test_pelt_penalty_scale_in_get_params():
    det = PELT(penalty_scale=2.5)
    assert det.get_params()["penalty_scale"] == 2.5


def test_pelt_penalty_scale_set_params_roundtrip():
    det = PELT()
    det.set_params(penalty_scale=3.0)
    assert det.penalty_scale == 3.0


def test_pelt_penalty_scale_clone():
    det = PELT(penalty_scale=1.8)
    cloned = clone(det)
    assert cloned.get_params()["penalty_scale"] == 1.8


def test_pelt_penalty_scale_multiplies_base_default_penalty():
    """PELT(penalty_scale=c) must give penalty_ = c * default_penalty."""
    X = _make_X()
    base = PELT(penalty_scale=1.0).fit(X)
    base_penalty = base.penalty_

    for c in [0.5, 2.0, 5.0]:
        det = PELT(penalty_scale=c).fit(X)
        assert det.penalty_ == pytest.approx(c * base_penalty, rel=1e-10)


def test_pelt_penalty_scale_multiplies_explicit_penalty():
    """With an explicit penalty, penalty_ = penalty_scale * penalty."""
    X = _make_X()
    for c in [0.5, 2.0]:
        det = PELT(penalty=4.0, penalty_scale=c).fit(X)
        assert det.penalty_ == pytest.approx(c * 4.0, rel=1e-10)


def test_pelt_penalty_scale_one_matches_legacy():
    """penalty_scale=1 must reproduce current behaviour (no-op)."""
    X = _make_X()
    old = PELT().fit(X)
    new = PELT(penalty_scale=1.0).fit(X)
    assert old.penalty_ == new.penalty_
    np.testing.assert_array_equal(
        old.predict_changepoints(X), new.predict_changepoints(X)
    )


def test_pelt_penalty_scale_changes_detections():
    """Large penalty_scale should suppress detections; small should increase them."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 1)), rng.normal(5, 1, (60, 1))])
    tight = PELT(penalty_scale=10.0).fit(X)
    loose = PELT(penalty_scale=0.1).fit(X)
    n_tight = len(tight.predict_changepoints(X))
    n_loose = len(loose.predict_changepoints(X))
    assert n_tight <= n_loose


# --------------------------------------------------------------------------- #
# _unpenalised_change_scores hook
# --------------------------------------------------------------------------- #


def test_pelt_unpenalised_change_scores_shape():
    """_unpenalised_change_scores returns a 1-D array."""
    X = _make_X()
    det = PELT().fit(X)
    S = det._unpenalised_change_scores(X)
    assert S.ndim == 1
    assert len(S) > 0


def test_pelt_unpenalised_change_scores_equals_brute_force():
    """S(τ) must equal cost(0,n) - cost(0,τ) - cost(τ,n) (brute-force check)."""
    X = _make_X(n=40, p=1)
    det = PELT().fit(X)
    n = X.shape[0]
    min_size = det.cost_.min_size

    cache = det.cost_.precompute(X)
    total = float(np.sum(det.cost_.evaluate(cache, np.array([[0, n]]))))

    taus = np.arange(min_size, n - min_size + 1)
    expected = np.empty(len(taus))
    for i, tau in enumerate(taus):
        left = float(np.sum(det.cost_.evaluate(cache, np.array([[0, tau]]))))
        right = float(np.sum(det.cost_.evaluate(cache, np.array([[tau, n]]))))
        expected[i] = total - left - right

    S = det._unpenalised_change_scores(X)
    np.testing.assert_allclose(S, expected, rtol=1e-10)


def test_pelt_unpenalised_change_scores_multivariate_sums_features():
    """Scores must be summed across features (not per-feature arrays)."""
    X = _make_X(n=40, p=3)
    det = PELT().fit(X)
    S = det._unpenalised_change_scores(X)
    assert S.ndim == 1


def test_pelt_unpenalised_change_scores_positive_on_change_data():
    """On data with a clear change, max(S) should be large and positive."""
    rng = np.random.default_rng(42)
    X = np.vstack([rng.normal(0, 0.1, (50, 1)), rng.normal(10, 0.1, (50, 1))])
    det = PELT().fit(X)
    S = det._unpenalised_change_scores(X)
    assert S.max() > 0.0
