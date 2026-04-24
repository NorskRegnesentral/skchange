"""Common contract tests for all interval scorers."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.conftest import make_no_change_X, make_single_change_X
from skchange.new_api.interval_scorers._base import (
    BaseIntervalScorer,
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
    valid = {"cost", "change_score", "saving", "transient_score"}
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


@_all_interval_scorers
def test_interval_scorer_precompute_wrong_n_features_raises(estimator):
    """precompute() with a mismatched number of features must raise ValueError."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    X_wrong = np.ones((X.shape[0], X.shape[1] + 1))
    with pytest.raises(ValueError):
        estimator.precompute(X_wrong)


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
    skchange_tags = estimator.__sklearn_tags__()
    tags = skchange_tags.interval_scorer_tags
    if tags.aggregated:
        assert scores.shape[1] == 1, "Aggregated scorer must return exactly 1 column."
    else:
        timestamps = skchange_tags.input_tags.timestamps
        n_signal_features = n_features - 1 if timestamps else n_features
        assert scores.shape[1] == n_signal_features, (
            f"Non-aggregated scorer must return 1 column per signal feature, "
            f"expected {n_signal_features}, got {scores.shape[1]}."
        )


@_all_interval_scorers
def test_interval_scorer_evaluate_output_range(estimator):
    """evaluate() must produce non-negative or finite scores for valid intervals."""
    tags = estimator.__sklearn_tags__().interval_scorer_tags
    X = make_no_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)
    scores = estimator.evaluate(cache, specs)
    atol = 1e-10
    if tags.non_negative_scores:
        assert np.all(scores >= -atol), (
            f"Interval scorer produced negative scores (tolerance={atol}): "
            f"min={scores.min():.6g}."
        )
    else:
        assert np.all(np.isfinite(scores)), (
            "Interval scorer produced non-finite scores: "
            f"min={scores.min():.6g}, max={scores.max():.6g}."
        )


@_all_interval_scorers
def test_interval_scorer_evaluate_output_dtype(estimator):
    """evaluate() must return floating-point values."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)
    scores = estimator.evaluate(cache, specs)
    assert np.issubdtype(
        scores.dtype, np.floating
    ), f"evaluate() must return float dtype, got {scores.dtype}."


@_all_interval_scorers
def test_interval_scorer_evaluate_empty_intervals(estimator):
    """evaluate() on empty interval_specs must return shape (0, k) without error."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    empty_specs = np.empty((0, estimator.interval_specs_ncols), dtype=np.int64)
    scores = estimator.evaluate(cache, empty_specs)
    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 2
    assert scores.shape[0] == 0


@_all_interval_scorers
def test_interval_scorer_evaluate_does_not_alter_cache(estimator):
    """evaluate() must not mutate the cache returned by precompute()."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    cache_snapshot = {
        k: v.copy() if isinstance(v, np.ndarray) else v for k, v in cache.items()
    }
    specs = make_interval_specs(estimator)
    estimator.evaluate(cache, specs)

    assert set(cache.keys()) == set(cache_snapshot.keys()), (
        f"evaluate() added or removed cache keys. "
        f"Before: {set(cache_snapshot.keys())}, after: {set(cache.keys())}."
    )
    for k, v_orig in cache_snapshot.items():
        v_now = cache[k]
        if isinstance(v_orig, np.ndarray):
            np.testing.assert_array_equal(
                v_now,
                v_orig,
                err_msg=f"Cache key {k!r} was mutated by evaluate().",
            )
        else:
            assert (
                v_now == v_orig
            ), f"Cache key {k!r} was mutated by evaluate(): {v_orig!r} -> {v_now!r}."


@_all_interval_scorers
def test_interval_scorer_evaluate_at_min_size(estimator):
    """evaluate() must work correctly for intervals of exactly min_size."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    min_s = estimator.min_size
    cache = estimator.precompute(X)
    ncols = estimator.interval_specs_ncols
    # Build minimum-size specs: each sub-interval is exactly min_s wide.
    if ncols == 2:
        specs = np.array([[0, min_s]], dtype=np.int64)
    elif ncols == 3:
        specs = np.array([[0, min_s, 2 * min_s]], dtype=np.int64)
    elif ncols == 4:
        specs = np.array([[0, min_s, 2 * min_s, 3 * min_s]], dtype=np.int64)
    else:
        pytest.skip(f"Unsupported interval_specs_ncols={ncols}")
    max_idx = specs.max()
    n = X.shape[0]
    if max_idx >= n:
        pytest.skip(
            f"Not enough samples ({n}) to form min_size specs (need {max_idx + 1})."
        )
    scores = estimator.evaluate(cache, specs)
    assert np.all(
        np.isfinite(scores)
    ), "evaluate() at min_size returned non-finite values."


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
def test_interval_scorer_get_default_penalty_before_fit_raises(estimator):
    """get_default_penalty() must raise NotFittedError before fit is called."""
    with pytest.raises(NotFittedError):
        estimator.get_default_penalty()


@_all_interval_scorers
def test_interval_scorer_get_default_penalty_positive(estimator):
    """get_default_penalty() must return positive values after fitting."""
    X = make_single_change_X(estimator)
    estimator.fit(X)
    penalty = estimator.get_default_penalty()
    assert isinstance(penalty, (int, float, np.ndarray))
    assert penalty > 0, f"Default penalty must be positive, got {penalty}."


# ---------------------------------------------------------------------------
# Statelessness tests
# ---------------------------------------------------------------------------


@_all_interval_scorers
def test_interval_scorer_precompute_does_not_alter_fitted_state(estimator):
    """precompute() must not modify the estimator's fitted attributes.

    Calling precompute() twice on different data should leave the estimator
    in exactly the same fitted state as after fit().
    """
    X = make_single_change_X(estimator)
    estimator.fit(X)

    fitted_state = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in estimator.__dict__.items()
    }

    # Call precompute twice with different data to stress-test statefulness.
    X2 = make_no_change_X(estimator)
    estimator.precompute(X)
    estimator.precompute(X2)

    for k, v_orig in fitted_state.items():
        v_now = estimator.__dict__[k]
        if isinstance(v_orig, np.ndarray):
            np.testing.assert_array_equal(
                v_now,
                v_orig,
                err_msg=f"Attribute {k!r} was modified by precompute().",
            )
        else:
            assert v_now == v_orig, (
                f"Attribute {k!r} was modified by precompute(): "
                f"{v_orig!r} -> {v_now!r}."
            )


@_all_interval_scorers
def test_interval_scorer_evaluate_does_not_alter_fitted_state(estimator):
    """evaluate() must not modify the estimator's fitted attributes.

    Calling evaluate() on different interval_specs using the same cache must
    leave the estimator's fitted state unchanged.
    """
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)

    fitted_state = {
        k: v.copy() if isinstance(v, np.ndarray) else v
        for k, v in estimator.__dict__.items()
    }

    specs = make_interval_specs(estimator, n_intervals=5)
    specs2 = make_interval_specs(estimator, n_intervals=3)
    estimator.evaluate(cache, specs)
    estimator.evaluate(cache, specs2)

    for k, v_orig in fitted_state.items():
        v_now = estimator.__dict__[k]
        if isinstance(v_orig, np.ndarray):
            np.testing.assert_array_equal(
                v_now,
                v_orig,
                err_msg=f"Attribute {k!r} was modified by evaluate().",
            )
        else:
            assert (
                v_now == v_orig
            ), f"Attribute {k!r} was modified by evaluate(): {v_orig!r} -> {v_now!r}."


@_all_interval_scorers
def test_interval_scorer_evaluate_same_result_on_repeated_calls(estimator):
    """evaluate() must return identical results when called repeatedly with the same
    cache and interval_specs (i.e. no hidden mutable state affects the output).
    """
    X = make_single_change_X(estimator)
    estimator.fit(X)
    cache = estimator.precompute(X)
    specs = make_interval_specs(estimator)

    scores1 = estimator.evaluate(cache, specs)
    scores2 = estimator.evaluate(cache, specs)

    np.testing.assert_array_equal(
        scores1,
        scores2,
        err_msg="evaluate() returned different results on repeated calls with the "
        "same cache and interval_specs.",
    )
