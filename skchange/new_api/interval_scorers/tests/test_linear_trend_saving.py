"""Tests for LinearTrendSaving with valued baseline parameters."""

import numpy as np

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import LinearTrendSaving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_SLOPE = 0.0
BASELINE_INTERCEPT = 0.0
SLOPE_AFTER = 1.0  # Slope of the second segment; should differ from BASELINE_SLOPE.


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_linear_trend_saving_stores_scalar_slope_and_intercept():
    """Scalar baseline params are stored as 1-D arrays after fit."""
    scorer = LinearTrendSaving(
        baseline_slope=BASELINE_SLOPE, baseline_intercept=BASELINE_INTERCEPT
    )
    X = make_no_change_X(scorer, loc=BASELINE_INTERCEPT)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_slope_, [BASELINE_SLOPE])
    np.testing.assert_array_equal(scorer.baseline_intercept_, [BASELINE_INTERCEPT])


def test_linear_trend_saving_broadcasts_to_n_features():
    """Scalar slope and intercept are broadcast to all signal features after fit."""
    scorer = LinearTrendSaving(baseline_slope=1.0, baseline_intercept=2.0)
    # LinearTrendSaving is not conditional; n_features=2 gives 2 signal features
    X = make_no_change_X(scorer, n_features=2, loc=2.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_slope_, [1.0, 1.0])
    np.testing.assert_array_equal(scorer.baseline_intercept_, [2.0, 2.0])


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_linear_trend_saving_finds_no_changepoint():
    """CAPA with a flat valued baseline finds no changepoints in stationary data."""
    scorer = LinearTrendSaving(
        baseline_slope=BASELINE_SLOPE, baseline_intercept=BASELINE_INTERCEPT
    )
    X = make_no_change_X(scorer, loc=BASELINE_INTERCEPT)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_linear_trend_saving_finds_single_changepoint():
    """CAPA with a flat valued baseline detects the slope change in kink data."""
    scorer = LinearTrendSaving(
        baseline_slope=BASELINE_SLOPE, baseline_intercept=BASELINE_INTERCEPT
    )
    X = make_single_change_X(
        scorer,
        linear_trend_slope_before=BASELINE_SLOPE,
        linear_trend_slope_after=SLOPE_AFTER,
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
