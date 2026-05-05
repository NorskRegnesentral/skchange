"""Tests for MultivariateTSaving with valued baseline parameters."""

import numpy as np

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import MultivariateTSaving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_MEAN = 0.0  # scalar; broadcast to all features
BASELINE_SCALE = 1.0  # scalar; broadcast to identity matrix
LOC_AFTER = 10.0  # Mean of the second segment; should differ from BASELINE_MEAN.


def _make_capa(saving):
    # min_segment_length=5 prevents spurious 2-3-sample detections from
    # random fluctuations in multivariate data.
    return CAPA(segment_saving=PenalisedScore(saving), min_segment_length=5)


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_multivariate_t_saving_stores_scalar_baseline():
    """Scalar mean and scale are broadcast to (p,) and (p,p) identity after fit."""
    scorer = MultivariateTSaving(
        baseline_mean=BASELINE_MEAN, baseline_scale=BASELINE_SCALE
    )
    X = make_no_change_X(scorer, n_features=2, loc=BASELINE_MEAN)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [BASELINE_MEAN, BASELINE_MEAN])
    np.testing.assert_array_almost_equal(
        scorer.baseline_scale_, BASELINE_SCALE * np.eye(2)
    )


def test_multivariate_t_saving_accepts_explicit_matrix():
    """An explicit SPD scale matrix is accepted and stored after fit."""
    scale = np.array([[2.0, 0.5], [0.5, 1.0]])
    scorer = MultivariateTSaving(
        baseline_mean=np.array([1.0, 2.0]), baseline_scale=scale
    )
    X = make_no_change_X(scorer, n_features=2, loc=1.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [1.0, 2.0])
    np.testing.assert_array_almost_equal(scorer.baseline_scale_, scale)


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_multivariate_t_saving_finds_no_changepoint():
    """CAPA with a matched multivariate-T valued baseline finds no changepoints."""
    scorer = MultivariateTSaving(
        baseline_mean=BASELINE_MEAN, baseline_scale=BASELINE_SCALE
    )
    X = make_no_change_X(scorer, n_features=2, loc=BASELINE_MEAN)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_multivariate_t_saving_finds_single_changepoint():
    """CAPA with a matched multivariate-T valued baseline detects the changepoint."""
    scorer = MultivariateTSaving(
        baseline_mean=BASELINE_MEAN, baseline_scale=BASELINE_SCALE
    )
    X = make_single_change_X(
        scorer, n_features=2, loc_before=BASELINE_MEAN, loc_after=LOC_AFTER
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
