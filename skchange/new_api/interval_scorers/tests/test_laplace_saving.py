"""Tests for LaplaceSaving with valued baseline parameters."""

import numpy as np
import pytest

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import LaplaceSaving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_LOCATION = 0.0
BASELINE_SCALE = 1.0
LOC_AFTER = 10.0  # Location of the second segment; should diff from BASELINE_LOCATION.


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_laplace_saving_stores_scalar_baseline():
    """Scalar baseline params are stored as 1-D arrays after fit."""
    scorer = LaplaceSaving(
        baseline_location=BASELINE_LOCATION, baseline_scale=BASELINE_SCALE
    )
    X = make_no_change_X(scorer, loc=BASELINE_LOCATION)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_location_, [BASELINE_LOCATION])
    np.testing.assert_array_equal(scorer.baseline_scale_, [BASELINE_SCALE])


def test_laplace_saving_broadcasts_to_n_features():
    """Scalar baselines are broadcast to all n_features after fit."""
    scorer = LaplaceSaving(baseline_location=2.0, baseline_scale=0.5)
    X = make_no_change_X(scorer, n_features=3, loc=2.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_location_, [2.0, 2.0, 2.0])
    np.testing.assert_array_equal(scorer.baseline_scale_, [0.5, 0.5, 0.5])


def test_laplace_saving_non_positive_scale_raises():
    """A non-positive baseline_scale must raise ValueError on fit."""
    scorer = LaplaceSaving(baseline_location=0.0, baseline_scale=-1.0)
    X = make_no_change_X(scorer, loc=0.0)
    with pytest.raises(ValueError):
        scorer.fit(X)


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_laplace_saving_finds_no_changepoint():
    """CAPA with a matched Laplace valued baseline finds no changepoints."""
    scorer = LaplaceSaving(
        baseline_location=BASELINE_LOCATION, baseline_scale=BASELINE_SCALE
    )
    X = make_no_change_X(scorer, loc=BASELINE_LOCATION)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_laplace_saving_finds_single_changepoint():
    """CAPA with a matched Laplace valued baseline detects the true changepoint."""
    scorer = LaplaceSaving(
        baseline_location=BASELINE_LOCATION, baseline_scale=BASELINE_SCALE
    )
    X = make_single_change_X(scorer, loc_before=BASELINE_LOCATION, loc_after=LOC_AFTER)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
