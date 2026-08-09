"""Tests for PELT.penalty_scale."""

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
    np.testing.assert_array_equal(old.predict(X), new.predict(X))


def test_pelt_penalty_scale_changes_detections():
    """Large penalty_scale should suppress detections; small should increase them."""
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (60, 1)), rng.normal(5, 1, (60, 1))])
    tight = PELT(penalty_scale=10.0).fit(X)
    loose = PELT(penalty_scale=0.1).fit(X)
    n_tight = len(tight.predict(X))
    n_loose = len(loose.predict(X))
    assert n_tight <= n_loose
