"""Dedicated tests for :class:`FPOP`.

Covers the sklearn compatibility checks that are skipped in the global
``test_sklearn_compatibility`` suite (see ``skchange/tests/test_all.py``)
because sklearn's checks pass multivariate data to ``fit``.  Tests here use
univariate data throughout.
"""

import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.detectors.tests._registry import _FPOP_INSTANCES

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)
X_UNIV = RNG.standard_normal((50, 1))
X_CHANGE = np.concatenate([RNG.normal(0, 1, (50, 1)), RNG.normal(5, 1, (50, 1))])


@pytest.fixture(params=_FPOP_INSTANCES, ids=repr)
def estimator(request):
    return clone(request.param)


# ---------------------------------------------------------------------------
# fit returns self
# ---------------------------------------------------------------------------


def test_fit_returns_self(estimator):
    assert estimator.fit(X_UNIV) is estimator


# ---------------------------------------------------------------------------
# check_is_fitted
# ---------------------------------------------------------------------------


def test_check_is_fitted(estimator):
    with pytest.raises(Exception):
        check_is_fitted(estimator)
    estimator.fit(X_UNIV)
    check_is_fitted(estimator)  # must not raise


# ---------------------------------------------------------------------------
# n_features_in_
# ---------------------------------------------------------------------------


def test_n_features_in_set_after_fit(estimator):
    estimator.fit(X_UNIV)
    assert estimator.n_features_in_ == 1


def test_predict_wrong_n_features_raises(estimator):
    estimator.fit(X_UNIV)
    X_multi = RNG.standard_normal((50, 3))
    with pytest.raises(ValueError):
        estimator.predict(X_multi)


# ---------------------------------------------------------------------------
# fit idempotent
# ---------------------------------------------------------------------------


def test_fit_idempotent(estimator):
    cpts1 = estimator.fit(X_CHANGE).predict(X_CHANGE)
    cpts2 = estimator.fit(X_CHANGE).predict(X_CHANGE)
    np.testing.assert_array_equal(cpts1, cpts2)


# ---------------------------------------------------------------------------
# single-observation predict returns empty array
# ---------------------------------------------------------------------------


def test_predict_single_observation_returns_empty(estimator):
    estimator.fit(X_UNIV)
    X_single = RNG.standard_normal((1, 1))
    cpts = estimator.predict(X_single)
    assert isinstance(cpts, np.ndarray)
    assert cpts.size == 0


# ---------------------------------------------------------------------------
# pickle round-trip
# ---------------------------------------------------------------------------


def test_pickle_roundtrip(estimator):
    estimator.fit(X_CHANGE)
    cpts_before = estimator.predict(X_CHANGE)
    restored = pickle.loads(pickle.dumps(estimator))
    cpts_after = restored.predict(X_CHANGE)
    np.testing.assert_array_equal(cpts_before, cpts_after)


# ---------------------------------------------------------------------------
# dtype handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_fit_predict_float_dtypes(estimator, dtype):
    X = X_CHANGE.astype(dtype)
    cpts = estimator.fit(X).predict(X)
    assert isinstance(cpts, np.ndarray)
    assert cpts.ndim == 1


# ---------------------------------------------------------------------------
# nan / inf input raises
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "-inf"]
)
def test_fit_raises_on_nan_inf(estimator, bad_value):
    X_bad = X_UNIV.copy()
    X_bad[5, 0] = bad_value
    with pytest.raises(ValueError):
        estimator.fit(X_bad)


# ---------------------------------------------------------------------------
# multivariate input raises at fit
# ---------------------------------------------------------------------------


def test_fit_rejects_multivariate(estimator):
    X_multi = RNG.standard_normal((50, 3))
    with pytest.raises(ValueError, match="univariate"):
        estimator.fit(X_multi)
