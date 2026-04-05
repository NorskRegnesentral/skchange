"""Common contract tests for all change detectors in ``skchange.new_api.detectors``."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.detectors.tests._registry import DETECTOR_TEST_INSTANCES

# `estimator` is a pytest fixture defined in conftest.py that clones each
# registry instance before passing it to the test, ensuring test isolation.
# indirect=True is needed to parametrize over the fixture rather than the raw instances.
_all_detectors = pytest.mark.parametrize(
    "estimator", DETECTOR_TEST_INSTANCES, indirect=True, ids=repr
)


# ---------------------------------------------------------------------------
# API contract tests (no fit required)
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_is_base_change_detector(estimator):
    """All registered detectors must inherit from BaseChangeDetector."""
    assert isinstance(estimator, BaseChangeDetector)


@_all_detectors
def test_detector_get_params_set_params(estimator):
    """get_params/set_params round-trip must be consistent (sklearn contract)."""
    got = estimator.get_params(deep=True)
    estimator.set_params(**got)
    assert estimator.get_params(deep=True) == got


@_all_detectors
def test_detector_clone(estimator):
    """clone() must produce an unfitted copy with identical params."""
    cloned = clone(estimator)
    assert type(cloned) is type(estimator)
    assert repr(cloned) == repr(estimator)


# ---------------------------------------------------------------------------
# fit() contract tests
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_fit_returns_self(estimator):
    """fit() must return self."""
    X = make_single_change_X(estimator)
    assert estimator.fit(X) is estimator


@_all_detectors
def test_detector_fit_sets_n_features_in(estimator):
    """fit() must set n_features_in_."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[1]


@_all_detectors
def test_detector_fit_sets_n_samples_in(estimator):
    """fit() must set n_samples_in_."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    assert hasattr(estimator, "n_samples_in_")
    assert estimator.n_samples_in_ == X.shape[0]


@_all_detectors
def test_detector_is_fitted_after_fit(estimator):
    """check_is_fitted() must pass after fit()."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    check_is_fitted(estimator)


@_all_detectors
def test_detector_fit_does_not_alter_input(estimator):
    """fit() must not mutate the input array."""
    X = make_single_change_X(estimator)
    X_copy = X.copy()
    estimator.fit(X)
    np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# predict_changepoints() contract tests
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_predict_changepoints_output_type(estimator):
    """predict_changepoints() must return a 1-D integer ndarray."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cpts = estimator.predict_changepoints(X)
    assert isinstance(cpts, np.ndarray)
    assert cpts.ndim == 1
    assert np.issubdtype(cpts.dtype, np.integer)


@_all_detectors
def test_detector_predict_changepoints_in_range(estimator):
    """All predicted changepoints must be valid sample indices."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cpts = estimator.predict_changepoints(X)
    assert np.all(cpts >= 1) and np.all(cpts <= X.shape[0] - 1)


@_all_detectors
def test_detector_predict_changepoints_sorted(estimator):
    """predict_changepoints() must return strictly sorted indices."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cpts = estimator.predict_changepoints(X)
    assert np.all(np.diff(cpts) > 0), "Changepoint indices must be strictly sorted."


# ---------------------------------------------------------------------------
# predict() contract tests
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_predict_output_shape(estimator):
    """predict() must return a 1-D integer array of length n_samples."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    labels = estimator.predict(X)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    assert np.issubdtype(labels.dtype, np.integer)


@_all_detectors
def test_detector_predict_labels_contiguous(estimator):
    """predict() labels must be 0-indexed and contiguous integers."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    labels = estimator.predict(X)
    unique = np.unique(labels)
    assert unique[0] == 0
    assert np.array_equal(unique, np.arange(len(unique)))


@_all_detectors
def test_detector_predict_consistent_with_predict_changepoints(estimator):
    """predict() segment boundaries must match predict_changepoints() indices."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    labels = estimator.predict(X)
    cpts = estimator.predict_changepoints(X)
    for cpt in cpts:
        assert (
            labels[cpt - 1] != labels[cpt]
        ), f"Expected label change at changepoint {cpt}."


# ---------------------------------------------------------------------------
# fit_predict() convenience test
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_fit_predict_equals_fit_then_predict(estimator):
    """fit_predict() must give the same result as fit() then predict()."""
    X = make_single_change_X(estimator)
    cloned = clone(estimator)
    labels_combined = estimator.fit_predict(X)
    labels_separate = cloned.fit(X).predict(X)
    np.testing.assert_array_equal(labels_combined, labels_separate)


# ---------------------------------------------------------------------------
# Simple detection sanity test
# ---------------------------------------------------------------------------


@_all_detectors
def test_detector_finds_single_changepoint(estimator):
    """Detectors must find the single changepoint in a simple two-segment problem."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cpts = estimator.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 5
    ), f"Changepoint {cpts[0]} is too far from the true changepoint {CHANGEPOINT}."


@_all_detectors
def test_detector_finds_no_changepoint(estimator):
    """Detectors must not flag changepoints in stationary (no-change) data."""
    X = make_no_change_X(estimator)
    estimator.fit(X)
    cpts = estimator.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"
