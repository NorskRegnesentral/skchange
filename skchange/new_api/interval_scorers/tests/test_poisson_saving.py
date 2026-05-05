"""Tests for PoissonSaving with valued baseline parameters."""

import numpy as np
import pytest

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import PenalisedScore, PoissonSaving

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_RATE = 2.0
RATE_AFTER = (
    10.0  # Rate of the second segment; should differ clearly from BASELINE_RATE.
)


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_poisson_saving_stores_scalar_baseline():
    """A scalar baseline_rate is stored as a 1-D array after fit."""
    scorer = PoissonSaving(baseline_rate=BASELINE_RATE)
    X = make_no_change_X(scorer, poisson_rate=BASELINE_RATE)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_rate_, [BASELINE_RATE])


def test_poisson_saving_broadcasts_to_n_features():
    """A scalar baseline_rate is broadcast to all n_features after fit."""
    scorer = PoissonSaving(baseline_rate=3.0)
    X = make_no_change_X(scorer, n_features=2, poisson_rate=3.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_rate_, [3.0, 3.0])


def test_poisson_saving_non_positive_rate_raises():
    """A non-positive baseline_rate must raise ValueError on fit."""
    scorer = PoissonSaving(baseline_rate=0.0)
    X = make_no_change_X(scorer, poisson_rate=2.0)
    with pytest.raises(ValueError):
        scorer.fit(X)


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_poisson_saving_finds_no_changepoint():
    """CAPA with a matched Poisson valued baseline finds no changepoints."""
    scorer = PoissonSaving(baseline_rate=BASELINE_RATE)
    X = make_no_change_X(scorer, poisson_rate=BASELINE_RATE)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_poisson_saving_finds_single_changepoint():
    """CAPA with a matched Poisson valued baseline detects the true changepoint."""
    scorer = PoissonSaving(baseline_rate=BASELINE_RATE)
    X = make_single_change_X(
        scorer, poisson_rate_before=BASELINE_RATE, poisson_rate_after=RATE_AFTER
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
