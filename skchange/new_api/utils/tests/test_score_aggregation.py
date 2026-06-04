"""Unit tests for ``skchange.new_api.utils._score_aggregation``."""

import numpy as np
import pytest

from skchange.new_api.interval_scorers._base import BaseIntervalScorer
from skchange.new_api.utils._score_aggregation import (
    aggregate_and_penalise,
    resolve_aggregation,
    resolve_penalty,
)
from skchange.new_api.utils._tags import IntervalScorerTags, SkchangeTags


class _StubScorer(BaseIntervalScorer):
    """Minimal scorer with configurable tags and default penalty.

    Used to exercise each branch of ``resolve_*`` / ``aggregate_and_penalise``
    without constructing real scorers and fitting them.
    """

    def __init__(
        self,
        *,
        aggregated: bool = False,
        penalised: bool = False,
        default_penalty: float | np.ndarray = 1.0,
    ):
        self.aggregated = aggregated
        self.penalised = penalised
        self.default_penalty = default_penalty

    def __sklearn_tags__(self):
        tags = SkchangeTags()
        tags.interval_scorer_tags = IntervalScorerTags()
        tags.interval_scorer_tags.aggregated = self.aggregated
        tags.interval_scorer_tags.penalised = self.penalised
        return tags

    def get_default_penalty(self):
        return self.default_penalty


# ---------------------------------------------------------------------------
# resolve_penalty
# ---------------------------------------------------------------------------


def test_resolve_penalty_returns_none_for_penalised_scorer():
    scorer = _StubScorer(penalised=True)
    assert resolve_penalty(scorer, None, 1.0, caller_name="Det") is None


def test_resolve_penalty_warns_when_user_penalty_with_penalised_scorer():
    scorer = _StubScorer(penalised=True)
    with pytest.warns(UserWarning):
        result = resolve_penalty(scorer, 10.0, 1.0, caller_name="Det")
    assert result is None


def test_resolve_penalty_warns_when_non_default_scale_with_penalised_scorer():
    scorer = _StubScorer(penalised=True)
    with pytest.warns(UserWarning):
        result = resolve_penalty(scorer, None, 2.0, caller_name="Det")
    assert result is None


def test_resolve_penalty_uses_default_when_penalty_is_none():
    default_penalty = 7.5
    scorer = _StubScorer(default_penalty=default_penalty)
    assert resolve_penalty(scorer, None, 1.0, caller_name="Det") == default_penalty


def test_resolve_penalty_applies_scale_to_default():
    default_penalty = 4.0
    penalty_scale = 2.0
    scorer = _StubScorer(default_penalty=default_penalty)
    out = resolve_penalty(scorer, None, penalty_scale, caller_name="Det")
    assert out == penalty_scale * default_penalty


def test_resolve_penalty_applies_scale_to_scalar_user_value():
    penalty = 3.0
    penalty_scale = 2.0
    scorer = _StubScorer()
    out = resolve_penalty(scorer, penalty, penalty_scale, caller_name="Det")
    assert out == penalty_scale * penalty


def test_resolve_penalty_applies_scale_to_array_user_value():
    penalty = np.array([1.0, 2.0, 3.0])
    penalty_scale = 2.0
    scorer = _StubScorer()
    out = resolve_penalty(scorer, penalty, penalty_scale, caller_name="Det")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, penalty_scale * penalty)


def test_resolve_penalty_rejects_negative_penalty():
    scorer = _StubScorer()
    with pytest.raises(ValueError):
        resolve_penalty(scorer, -1.0, 1.0, caller_name="Det")


def test_resolve_penalty_rejects_non_decreasing_array():
    scorer = _StubScorer()
    with pytest.raises(ValueError):
        resolve_penalty(scorer, np.array([3.0, 1.0]), 1.0, caller_name="Det")


# ---------------------------------------------------------------------------
# resolve_aggregation
# ---------------------------------------------------------------------------


def test_resolve_aggregation_passthrough_for_penalised_scorer():
    scorer = _StubScorer(penalised=True)
    mode = resolve_aggregation(scorer, "sum", None, n_features=3, caller_name="Det")
    assert mode == "passthrough"


def test_resolve_aggregation_passthrough_warns_for_non_default_agg():
    scorer = _StubScorer(penalised=True)
    with pytest.warns(UserWarning):
        mode = resolve_aggregation(scorer, "max", None, n_features=3, caller_name="Det")
    assert mode == "passthrough"


def test_resolve_aggregation_passthrough_silent_without_caller_name():
    scorer = _StubScorer(penalised=True)
    with warnings_as_errors():
        mode = resolve_aggregation(scorer, "max", None, n_features=3)
    assert mode == "passthrough"


def test_resolve_aggregation_scalar_penalty_respects_agg_sum():
    scorer = _StubScorer()
    mode = resolve_aggregation(scorer, "sum", 1.0, n_features=3, caller_name="Det")
    assert mode == "sum"


def test_resolve_aggregation_scalar_penalty_respects_agg_max():
    scorer = _StubScorer()
    mode = resolve_aggregation(scorer, "max", 1.0, n_features=3, caller_name="Det")
    assert mode == "max"


def test_resolve_aggregation_array_penalty_with_linear_diffs_is_top_k_linear():
    scorer = _StubScorer()
    penalty = np.array([1.0, 2.0, 3.0])
    mode = resolve_aggregation(scorer, "sum", penalty, n_features=3, caller_name="Det")
    assert mode == "top_k_linear"


def test_resolve_aggregation_array_penalty_with_nonlinear_diffs_is_top_k_nonlinear():
    scorer = _StubScorer()
    penalty = np.array([1.0, 2.0, 5.0])
    mode = resolve_aggregation(scorer, "sum", penalty, n_features=3, caller_name="Det")
    assert mode == "top_k_nonlinear"


def test_resolve_aggregation_array_penalty_with_agg_max_raises():
    scorer = _StubScorer()
    penalty = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        resolve_aggregation(scorer, "max", penalty, n_features=3, caller_name="Det")


def test_resolve_aggregation_array_penalty_wrong_length_raises():
    scorer = _StubScorer()
    penalty = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        resolve_aggregation(scorer, "sum", penalty, n_features=3, caller_name="Det")


def test_resolve_aggregation_aggregated_scorer_with_array_penalty_raises():
    scorer = _StubScorer(aggregated=True)
    penalty = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        resolve_aggregation(scorer, "sum", penalty, n_features=3, caller_name="Det")


def test_resolve_aggregation_aggregated_scorer_with_non_default_agg_warns():
    scorer = _StubScorer(aggregated=True)
    with pytest.warns(UserWarning):
        mode = resolve_aggregation(scorer, "max", 1.0, n_features=3, caller_name="Det")
    # The mode follows the user choice; aggregate_and_penalise ignores agg
    # for already-aggregated scorers via the (n_intervals, 1) shape — the
    # warning exists to flag the inconsistency to the user.
    assert mode == "max"


def test_resolve_aggregation_aggregated_scorer_silent_without_caller_name():
    scorer = _StubScorer(aggregated=True)
    with warnings_as_errors():
        mode = resolve_aggregation(scorer, "max", 1.0, n_features=3)
    assert mode == "max"


# ---------------------------------------------------------------------------
# aggregate_and_penalise
# ---------------------------------------------------------------------------


def test_aggregate_and_penalise_sum():
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    penalty = 1.0
    out = aggregate_and_penalise(raw, "sum", penalty=penalty)
    np.testing.assert_array_equal(out, raw.sum(axis=1) - penalty)


def test_aggregate_and_penalise_max():
    raw = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    penalty = 1.0
    out = aggregate_and_penalise(raw, "max", penalty=penalty)
    np.testing.assert_array_equal(out, raw.max(axis=1) - penalty)


def test_aggregate_and_penalise_top_k_linear_matches_top_k_nonlinear():
    """For a linear penalty the two top-k modes must agree."""
    rng = np.random.default_rng(0)
    # Data must be > penalty[0] to avoid zeroing out all scores in the linear penalty
    # version. In this case, the two aggregations are not identical. See the CAPA paper.
    raw = 5.0 + rng.random((20, 4))
    penalty = np.array([1.0, 2.0, 3.0, 4.0])
    linear = aggregate_and_penalise(raw, "top_k_linear", penalty)
    nonlinear = aggregate_and_penalise(raw, "top_k_nonlinear", penalty)
    np.testing.assert_allclose(linear, nonlinear)


def test_aggregate_and_penalise_passthrough_flattens_single_column():
    raw = np.array([[1.0], [2.0], [3.0]])
    out = aggregate_and_penalise(raw, "passthrough", penalty=None)
    np.testing.assert_array_equal(out, raw.reshape(-1))
    assert out.ndim == 1


def test_aggregate_and_penalise_unknown_mode_raises():
    raw = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        aggregate_and_penalise(raw, "bogus", penalty=0.0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class warnings_as_errors:
    """Context manager that fails the test if any warning is raised."""

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings()
        self._cm.__enter__()
        warnings.simplefilter("error")
        return self

    def __exit__(self, *args):
        return self._cm.__exit__(*args)
