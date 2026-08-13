"""Tests for BaseChangeDetector."""

import numpy as np
import pytest
from sklearn.utils import Tags

from skchange.new_api.detectors._base import BaseChangeDetector
from skchange.new_api.utils._tags import ChangeDetectorTags, SkchangeTags


class _StubDetector(BaseChangeDetector):
    """Minimal concrete detector for testing the base class."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_samples_in_ = X.shape[0]
        return self

    def predict(self, X):
        return np.array(self._changepoints, dtype=np.intp)

    def set_changepoints(self, changepoints):
        self._changepoints = changepoints
        return self


_N = 10


@pytest.fixture
def stub():
    det = _StubDetector()
    det.fit(np.zeros((_N, 1)))
    return det


def test_predict_not_implemented():
    """predict() must raise NotImplementedError on the bare base class."""

    class BareDetector(BaseChangeDetector):
        def fit(self, X, y=None):
            return self

    with pytest.raises(NotImplementedError):
        BareDetector().predict(np.zeros((10, 1)))


def test_predict_no_changepoints(stub):
    """predict() with no changepoints returns an empty array."""
    stub.set_changepoints([])
    cpts = stub.predict(np.zeros((_N, 1)))
    assert cpts.shape == (0,)


def test_predict_returns_changepoints(stub):
    """predict() must return the changepoint indices."""
    stub.set_changepoints([4, 7])
    cpts = stub.predict(np.zeros((_N, 1)))
    np.testing.assert_array_equal(cpts, np.array([4, 7]))


def test_fit_predict_equals_fit_then_predict():
    """fit_predict() must give the same result as fit().predict()."""
    X = np.zeros((_N, 1))
    det1 = _StubDetector().set_changepoints([3, 7])
    det2 = _StubDetector().set_changepoints([3, 7])
    cpts_combined = det1.fit_predict(X)
    cpts_separate = det2.fit(X).predict(X)
    np.testing.assert_array_equal(cpts_combined, cpts_separate)


def test_sklearn_tags_type():
    """__sklearn_tags__() must return a SkchangeTags instance."""
    tags = _StubDetector().__sklearn_tags__()
    assert isinstance(tags, Tags)
    assert isinstance(tags, SkchangeTags)


def test_sklearn_tags_has_change_detector_tags():
    """__sklearn_tags__() must set change_detector_tags."""
    tags = _StubDetector().__sklearn_tags__()
    assert isinstance(tags.change_detector_tags, ChangeDetectorTags)
