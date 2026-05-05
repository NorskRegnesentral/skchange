"""Tests for LinearRegressionSaving with valued baseline parameters."""

import numpy as np

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import LinearRegressionSaving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_COEF = 1.0
COEF_AFTER = 5.0  # Coefficient of the second segment; should differ from BASELINE_COEF.


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_linear_regression_saving_stores_scalar_baseline():
    """A scalar baseline_coeffs is broadcast to n_covariates and stored after fit."""
    scorer = LinearRegressionSaving(baseline_coeffs=BASELINE_COEF)
    X = make_no_change_X(scorer, regression_coef=BASELINE_COEF)
    scorer.fit(X)
    # n_features=2 by default for conditional; n_covariates = 1
    np.testing.assert_array_equal(scorer.baseline_coeffs_, [BASELINE_COEF])


def test_linear_regression_saving_length1_array_broadcasts():
    """A length-1 baseline_coeffs array is broadcast to n_covariates after fit."""
    scorer = LinearRegressionSaving(baseline_coeffs=np.array([2.0]))
    X = make_no_change_X(scorer, n_features=4, regression_coef=2.0)
    scorer.fit(X)
    # n_features=4 → n_covariates=3; scalar 2.0 broadcast to all three
    np.testing.assert_array_equal(scorer.baseline_coeffs_, [2.0, 2.0, 2.0])


def test_linear_regression_saving_array_stored_as_is():
    """A full-length baseline_coeffs array is stored unchanged after fit."""
    scorer = LinearRegressionSaving(baseline_coeffs=np.array([1.0, 2.0]))
    X = make_no_change_X(scorer, n_features=3, regression_coef=1.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_coeffs_, [1.0, 2.0])


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_linear_regression_saving_finds_no_changepoint():
    """CAPA with a matched regression valued baseline finds no changepoints."""
    scorer = LinearRegressionSaving(baseline_coeffs=BASELINE_COEF)
    X = make_no_change_X(scorer, regression_coef=BASELINE_COEF)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_linear_regression_saving_finds_single_changepoint():
    """CAPA with a matched regression valued baseline detects the true changepoint."""
    scorer = LinearRegressionSaving(baseline_coeffs=BASELINE_COEF)
    X = make_single_change_X(
        scorer,
        regression_coef_before=BASELINE_COEF,
        regression_coef_after=COEF_AFTER,
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
