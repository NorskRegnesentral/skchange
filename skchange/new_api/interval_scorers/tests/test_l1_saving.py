"""Tests for L1Saving with valued baseline parameters."""

import numpy as np

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import L1Saving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_LOCATION = 0.0
LOC_AFTER = (
    10.0  # Location of the second segment; should differ from BASELINE_LOCATION.
)


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_l1_saving_stores_scalar_baseline():
    """A scalar baseline_location is stored as a 1-D array after fit."""
    scorer = L1Saving(baseline_location=BASELINE_LOCATION)
    X = make_no_change_X(scorer, loc=BASELINE_LOCATION)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_location_, [BASELINE_LOCATION])


def test_l1_saving_broadcasts_to_n_features():
    """A scalar baseline_location is broadcast to all n_features after fit."""
    scorer = L1Saving(baseline_location=3.0)
    X = make_no_change_X(scorer, n_features=3, loc=3.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_location_, [3.0, 3.0, 3.0])


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_l1_saving_finds_no_changepoint():
    """CAPA with a matched L1 valued baseline finds no changepoints."""
    scorer = L1Saving(baseline_location=BASELINE_LOCATION)
    X = make_no_change_X(scorer, loc=BASELINE_LOCATION)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_l1_saving_finds_single_changepoint():
    """CAPA with a matched L1 valued baseline detects the true changepoint."""
    scorer = L1Saving(baseline_location=BASELINE_LOCATION)
    X = make_single_change_X(scorer, loc_before=BASELINE_LOCATION, loc_after=LOC_AFTER)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
