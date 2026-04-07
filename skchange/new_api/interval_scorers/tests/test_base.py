"""Tests for BaseIntervalScorer and its convenience subclasses."""

import numpy as np
import pytest

from skchange.new_api.interval_scorers._base import (
    BaseChangeScore,
    BaseCost,
    BaseIntervalScorer,
    BaseSaving,
    BaseTransientScore,
)
from skchange.new_api.utils._tags import IntervalScorerTags, SkchangeTags


class _StubScorer(BaseIntervalScorer):
    """Minimal concrete scorer that uses the default precompute and min_size."""

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.n_samples_in_ = X.shape[0]
        return self

    def evaluate(self, cache, interval_specs):
        return np.zeros((len(interval_specs), self.n_features_in_))

    @property
    def interval_specs_ncols(self):
        return 2


_X = np.random.default_rng(0).normal(size=(50, 2))


# ---------------------------------------------------------------------------
# Abstract method / NotImplementedError tests
# ---------------------------------------------------------------------------


def test_fit_not_implemented():
    class Bare(BaseIntervalScorer):
        pass

    with pytest.raises(NotImplementedError):
        Bare().fit(_X)


def test_evaluate_not_implemented():
    class Bare(BaseIntervalScorer):
        def fit(self, X, y=None):
            self.n_features_in_ = X.shape[1]
            self.n_samples_in_ = X.shape[0]
            return self

    scorer = Bare().fit(_X)
    with pytest.raises(NotImplementedError):
        scorer.evaluate({}, np.array([[0, 10]]))


def test_interval_specs_ncols_not_implemented():
    class Bare(BaseIntervalScorer):
        def fit(self, X, y=None):
            return self

        def evaluate(self, cache, specs):
            return np.zeros(len(specs))

    with pytest.raises(NotImplementedError):
        _ = Bare().interval_specs_ncols


# ---------------------------------------------------------------------------
# Default precompute() behaviour
# ---------------------------------------------------------------------------


def test_default_precompute_returns_dict_with_X():
    scorer = _StubScorer().fit(_X)
    cache = scorer.precompute(_X)
    assert isinstance(cache, dict)
    assert "X" in cache
    assert isinstance(cache["X"], np.ndarray)


def test_default_precompute_requires_fitted():
    with pytest.raises(Exception):
        _StubScorer().precompute(_X)


# ---------------------------------------------------------------------------
# Default min_size
# ---------------------------------------------------------------------------


def test_default_min_size_is_one():
    assert _StubScorer().min_size == 1


# ---------------------------------------------------------------------------
# Default get_default_penalty()
# ---------------------------------------------------------------------------


def test_get_default_penalty_requires_fitted():
    with pytest.raises(Exception):
        _StubScorer().get_default_penalty()


def test_get_default_penalty_positive_after_fit():
    scorer = _StubScorer().fit(_X)
    penalty = scorer.get_default_penalty()
    assert isinstance(penalty, (int, float, np.ndarray))
    assert np.all(penalty > 0)


# ---------------------------------------------------------------------------
# Default __sklearn_tags__()
# ---------------------------------------------------------------------------


def test_base_sklearn_tags_has_interval_scorer_tags():
    tags = _StubScorer().__sklearn_tags__()
    assert isinstance(tags, SkchangeTags)
    assert isinstance(tags.interval_scorer_tags, IntervalScorerTags)


# ---------------------------------------------------------------------------
# Convenience subclasses: score_type and interval_specs_ncols
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base_cls, expected_score_type, expected_ncols",
    [
        (BaseCost, "cost", 2),
        (BaseChangeScore, "change_score", 3),
        (BaseSaving, "saving", 2),
        (BaseTransientScore, "transient_score", 4),
    ],
)
def test_convenience_base_tags(base_cls, expected_score_type, expected_ncols):
    class Concrete(base_cls):
        def fit(self, X, y=None):
            return self

        def evaluate(self, cache, specs):
            return np.zeros(len(specs))

    scorer = Concrete()
    tags = scorer.__sklearn_tags__().interval_scorer_tags
    assert tags.score_type == expected_score_type
    assert scorer.interval_specs_ncols == expected_ncols
