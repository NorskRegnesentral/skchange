"""Common contract tests for all interval scorers."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.conftest import make_no_change_X, make_single_change_X
from skchange.new_api.interval_scorers._base import (
    BaseIntervalScorer,
    is_penalised_score,
)
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)
from skchange.new_api.utils._tags import IntervalScorerTags

# `estimator` is a pytest fixture defined in conftest.py that clones each
# registry instance before passing it to the test, ensuring test isolation.
_all_interval_scorers = pytest.mark.parametrize(
    "estimator", INTERVAL_SCORER_TEST_INSTANCES, indirect=True, ids=repr
)


def make_interval_specs(
    estimator: BaseIntervalScorer, n_intervals: int = 5
) -> np.ndarray:
    """Generate valid interval_specs for the given scorer based on its ncols."""
    ncols = estimator.interval_specs_ncols
    n = 50
    step = n // (n_intervals + 1)
    rows = []
    for i in range(n_intervals):
        start = i * step
        end = start + step
        if ncols == 2:
            rows.append([start, end])
        elif ncols == 3:
            split = (start + end) // 2
            rows.append([start, split, end])
        elif ncols == 4:
            quarter = (end - start) // 4
            inner_start = start + quarter
            inner_end = end - quarter
            rows.append([start, inner_start, inner_end, end])
        else:
            raise NotImplementedError(f"Unsupported interval_specs_ncols={ncols}")
    return np.array(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# API contract tests (no fit required)
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_is_base_interval_scorer(estimator):
    """All registered scorers must inherit from BaseIntervalScorer."""
    assert isinstance(estimator, BaseIntervalScorer)


@_all_interval_scorers
def test_interval_scorer_get_params_set_params(estimator):
    """get_params/set_params round-trip must be consistent (sklearn contract)."""
    got = estimator.get_params(deep=True)
    estimator.set_params(**got)
    assert estimator.get_params(deep=True) == got


@_all_interval_scorers
def test_interval_scorer_clone(estimator):
    """clone() must produce an unfitted copy with identical repr."""
    cloned = clone(estimator)
    assert type(cloned) is type(estimator)
    assert repr(cloned) == repr(estimator)


@_all_interval_scorers
def test_interval_scorer_sklearn_tags_type(estimator):
    """__sklearn_tags__() must return an object with interval_scorer_tags set."""
    tags = estimator.__sklearn_tags__()
    assert isinstance(tags.interval_scorer_tags, IntervalScorerTags)


@_all_interval_scorers
def test_interval_scorer_score_type_valid(estimator):
    """score_type tag must be one of the recognised values."""
    valid = {"cost", "change_score", "saving", "local_saving", "transient_score"}
    score_type = estimator.__sklearn_tags__().interval_scorer_tags.score_type
    assert score_type in valid, f"Unexpected score_type: {score_type!r}"


# ---------------------------------------------------------------------------
# fit() contract tests
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_fit_returns_self(estimator):
    """fit() must return self."""
    X = make_single_change_X(estimator)
    assert estimator.fit(X) is estimator


@_all_interval_scorers
def test_interval_scorer_fit_sets_n_features_in(estimator):
    """fit() must set n_features_in_."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[1]


@_all_interval_scorers
def test_interval_scorer_fit_sets_n_samples_in(estimator):
    """fit() must set n_samples_in_."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    assert hasattr(estimator, "n_samples_in_")
    assert estimator.n_samples_in_ == X.shape[0]


@_all_interval_scorers
def test_interval_scorer_is_fitted_after_fit(estimator):
    """check_is_fitted() must pass after fit()."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    check_is_fitted(estimator)


@_all_interval_scorers
def test_interval_scorer_fit_does_not_alter_input(estimator):
    """fit() must not mutate the input array."""
    X = make_single_change_X(estimator)
    X_copy = X.copy()
    estimator.fit(X)
    np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# precompute() contract tests
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_precompute_returns_dict(estimator):
    """precompute() must return a dict."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    assert isinstance(cache, dict)


@_all_interval_scorers
def test_interval_scorer_precompute_does_not_alter_input(estimator):
    """precompute() must not mutate the input array."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    X_copy = X.copy()
    estimator.precompute(X)
    np.testing.assert_array_equal(X, X_copy)


# ---------------------------------------------------------------------------
# evaluate() contract tests
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_evaluate_output_shape(estimator):
    """evaluate() must return a 2d array with a prescribed number of columns."""
    # Use 2-feature data so non-aggregated scorers have a meaningful column count to
    # assert.
    n_features = 2
    X = make_single_change_X(estimator, n_features=n_features)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)
    scores = estimator.evaluate(cache, specs)
    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 2, "evaluate() must return a 2-D array."
    assert scores.shape[0] == len(specs)
    tags = estimator.__sklearn_tags__().interval_scorer_tags
    if tags.aggregated:
        assert scores.shape[1] == 1, "Aggregated scorer must return exactly 1 column."
    else:
        assert scores.shape[1] == n_features, (
            f"Non-aggregated scorer must return 1 column per feature, "
            f"expected {n_features}, got {scores.shape[1]}."
        )


@_all_interval_scorers
def test_interval_scorer_evaluate_output_finite(estimator):
    """evaluate() must return only finite values for valid intervals."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)
    scores = estimator.evaluate(cache, specs)
    assert np.all(np.isfinite(scores)), "evaluate() returned non-finite values."


# ---------------------------------------------------------------------------
# interval_specs_ncols property test
# ---------------------------------------------------------------------------
@_all_interval_scorers
def test_interval_scorer_interval_specs_ncols(estimator):
    """interval_specs_ncols must be a positive integer (requires fit for wrappers)."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    ncols = estimator.interval_specs_ncols
    assert isinstance(ncols, int) and ncols >= 1


# ---------------------------------------------------------------------------
# min_size property test
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_min_size(estimator):
    """min_size must be a positive integer."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    assert isinstance(estimator.min_size, int) and estimator.min_size >= 1


# ---------------------------------------------------------------------------
# get_default_penalty() test
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_get_default_penalty_positive(estimator):
    """get_default_penalty() must return a positive scalar after fitting."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    penalty = estimator.get_default_penalty()
    assert isinstance(penalty, (int, float))
    assert penalty > 0, f"Default penalty must be positive, got {penalty}."


# ---------------------------------------------------------------------------
# Sanity tests
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scores_non_negative(estimator):
    """Interval scorers must produce non-negative scores on representative intervals."""
    if is_penalised_score(estimator):
        pytest.skip("Only for unpenalised scorers.")
    X = make_no_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)
    scores = estimator.evaluate(cache, specs)
    atol = 1e-10
    assert np.all(scores >= -atol), (
        f"Interval scorer produced negative scores (tolerance={atol}): "
        f"min={scores.min():.6g}."
    )
