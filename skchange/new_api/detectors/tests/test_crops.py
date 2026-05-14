"""Dedicated tests for :class:`CROPS` and its helpers.

These tests target branches not exercised by the generic registry-based
contract tests in ``test_all.py``: input validation, internal helper
functions, the elbow / BIC selection paths, and edge cases in the CROPS
search loop.
"""

import numpy as np
import pytest

from skchange.new_api.detectors._crops import (
    CROPS,
    _crops_elbow_scores,
    _evaluate_segmentation,
    _segmentation_bic_value,
)
from skchange.new_api.interval_scorers import GaussianCost, L2Cost, PenalisedScore


def _make_data(n_per_seg: int = 60, n_segments: int = 3, seed: int = 0) -> np.ndarray:
    """Generate piecewise-constant Gaussian data with ``n_segments`` segments.

    True changepoints are at multiples of ``n_per_seg``.
    """
    rng = np.random.default_rng(seed)
    means = np.arange(n_segments) * 10.0
    parts = [rng.normal(loc=m, scale=1.0, size=(n_per_seg, 1)) for m in means]
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_evaluate_segmentation_no_changepoints():
    """With no changepoints the cost is the cost of the full interval."""
    X = _make_data()
    cost = L2Cost().fit(X)
    cache = cost.precompute(X)
    n = X.shape[0]
    expected = float(np.sum(cost.evaluate(cache, np.array([[0, n]]))))
    got = _evaluate_segmentation(cost, cache, np.array([], dtype=np.intp), n)
    assert got == pytest.approx(expected)


def test_evaluate_segmentation_accepts_2d_changepoints():
    """A 2D changepoints array is flattened internally (PELT output is 2D)."""
    X = _make_data()
    cost = L2Cost().fit(X)
    cache = cost.precompute(X)
    n = X.shape[0]
    cps_1d = np.array([60, 120], dtype=np.intp)
    cps_2d = cps_1d.reshape(-1, 1)
    assert _evaluate_segmentation(cost, cache, cps_2d, n) == pytest.approx(
        _evaluate_segmentation(cost, cache, cps_1d, n)
    )


def test_evaluate_segmentation_rejects_non_monotonic():
    """Non-strictly-increasing changepoints raise ``ValueError``."""
    X = _make_data()
    cost = L2Cost().fit(X)
    cache = cost.precompute(X)
    with pytest.raises(ValueError, match="strictly increasing"):
        _evaluate_segmentation(
            cost, cache, np.array([60, 60], dtype=np.intp), X.shape[0]
        )


def test_segmentation_bic_value_matches_formula():
    """BIC value equals ``seg_cost + n_segments * per_segment_penalty``."""
    X = _make_data()
    cost = L2Cost().fit(X)
    cache = cost.precompute(X)
    n = X.shape[0]
    cps = np.array([60, 120], dtype=np.intp)
    seg_cost = _evaluate_segmentation(cost, cache, cps, n)
    per_seg_pen = float(np.sum(cost.get_default_penalty()))
    expected = seg_cost + (len(cps) + 1) * per_seg_pen
    assert _segmentation_bic_value(cost, cache, cps, n) == pytest.approx(expected)


def test_crops_elbow_scores_too_few_segmentations_warns():
    """Fewer than 3 segmentations triggers a warning and returns all ``-inf``."""
    with pytest.warns(UserWarning, match="Not enough segmentations"):
        scores = _crops_elbow_scores(np.array([0, 1]), np.array([10.0, 5.0]))
    assert scores.shape == (2,)
    assert np.all(np.isneginf(scores))


def test_crops_elbow_scores_endpoints_are_neginf():
    """First and last elbow scores are always ``-inf``; interior is finite."""
    scores = _crops_elbow_scores(
        np.array([0, 1, 2, 3]), np.array([100.0, 50.0, 20.0, 10.0])
    )
    assert np.isneginf(scores[0])
    assert np.isneginf(scores[-1])
    assert np.all(np.isfinite(scores[1:-1]))


# ---------------------------------------------------------------------------
# CROPS.fit input validation
# ---------------------------------------------------------------------------


def test_fit_rejects_too_few_samples():
    """Fitting on fewer than ``2 * cost.min_size`` samples raises."""
    X = np.zeros((1, 1))
    with pytest.raises(ValueError, match="at least 2 \\* cost.min_size"):
        CROPS().fit(X)


def test_fit_rejects_min_penalty_geq_max_penalty():
    """``min_penalty >= max_penalty`` raises a ``ValueError``."""
    X = _make_data()
    with pytest.raises(ValueError, match="strictly less than"):
        CROPS(min_penalty=10.0, max_penalty=10.0).fit(X)


def test_fit_rejects_min_segment_length_below_cost_min_size():
    """``min_segment_length`` below ``cost.min_size`` is rejected."""
    X = _make_data()
    with pytest.raises(ValueError, match="at least"):
        CROPS(cost=GaussianCost(), min_segment_length=1).fit(X)


def test_fit_rejects_min_segment_length_above_step_size():
    """``min_segment_length`` cannot exceed ``step_size`` when ``step_size > 1``."""
    X = _make_data()
    with pytest.raises(ValueError, match="cannot be"):
        CROPS(step_size=3, min_segment_length=5).fit(X)


def test_fit_rejects_penalised_score():
    """Passing a ``PenalisedScore`` as ``cost`` is rejected."""
    X = _make_data()
    with pytest.raises(ValueError):
        CROPS(cost=PenalisedScore(L2Cost())).fit(X)


# ---------------------------------------------------------------------------
# CROPS prediction paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("selection_method", ["bic", "elbow"])
def test_predict_all_metadata_contract(selection_method):
    """``predict_all`` exposes the full path metadata and lookup table."""
    X = _make_data(n_segments=4)
    detector = CROPS(
        selection_method=selection_method, min_penalty=0.5, max_penalty=200.0
    ).fit(X)
    result = detector.predict_all(X)

    assert set(result) == {
        "changepoints",
        "changepoints_metadata",
        "changepoints_lookup",
        "optimal_penalty",
    }
    md = result["changepoints_metadata"]
    for key in ("num_changepoints", "penalty", "segmentation_cost", "optimum_value"):
        assert key in md
    score_key = "bic_value" if selection_method == "bic" else "elbow_score"
    assert score_key in md
    # num_changepoints sorted ascending.
    assert np.all(np.diff(md["num_changepoints"]) > 0)
    # Lookup matches metadata entries.
    for n_cp in md["num_changepoints"]:
        assert int(n_cp) in result["changepoints_lookup"]


def test_predict_changepoints_recovers_true_changepoints():
    """CROPS with BIC selection recovers the true changepoints on clean data."""
    X = _make_data(n_per_seg=60, n_segments=3)
    cps = CROPS().fit(X).predict_changepoints(X)
    np.testing.assert_array_equal(cps, [60, 120])


def test_elbow_selection_with_short_path_warns():
    """When the CROPS path has fewer than 3 entries, elbow selection warns."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 1))  # no-change data; only one segmentation
    detector = CROPS(selection_method="elbow", min_penalty=1e3, max_penalty=1e4).fit(X)
    with pytest.warns(UserWarning, match="Not enough segmentations"):
        detector.predict_all(X)


def test_prune_false_matches_prune_true():
    """Disabling pruning yields the same optimal changepoints."""
    X = _make_data()
    cps_pruned = CROPS(min_penalty=0.5, max_penalty=200.0, prune=True).fit_predict(X)
    cps_unpruned = (
        CROPS(min_penalty=0.5, max_penalty=200.0, prune=False)
        .fit(X)
        .predict_changepoints(X)
    )
    np.testing.assert_array_equal(cps_pruned, cps_unpruned)


def test_default_penalty_bounds_resolved_from_cost():
    """When ``min_penalty``/``max_penalty`` are ``None``, defaults derive from cost."""
    X = _make_data()
    detector = CROPS().fit(X)
    default_pen = float(np.sum(detector.cost_.get_default_penalty()))
    assert detector.min_penalty_ == pytest.approx(0.5 * default_pen)
    assert detector.max_penalty_ == pytest.approx(5.0 * default_pen)


def test_step_size_greater_than_one_constrains_changepoints():
    """``step_size > 1`` exercises the step-size PELT path; cps respect the step."""
    X = _make_data()
    cps = CROPS(step_size=5).fit(X).predict_changepoints(X)
    assert np.all(cps % 5 == 0)


def test_min_segment_length_one_recovers_changepoints():
    """``min_segment_length=1`` exercises the dedicated PELT-min-1 path."""
    X = _make_data(n_per_seg=60, n_segments=3)
    cps = CROPS(min_segment_length=1).fit(X).predict_changepoints(X)
    np.testing.assert_array_equal(cps, [60, 120])
