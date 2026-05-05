"""Tests for GaussianSaving with valued baseline parameters."""

import numpy as np
import pytest

from skchange.new_api.conftest import (
    CHANGEPOINT,
    make_no_change_X,
    make_single_change_X,
)
from skchange.new_api.detectors import CAPA
from skchange.new_api.interval_scorers import GaussianSaving, PenalisedScore

# Baseline parameters — must match data-generation parameters in the sanity tests.
BASELINE_MEAN = 0.0
BASELINE_VAR = 1.0
LOC_AFTER = (
    10.0  # Mean of the second segment; should differ clearly from BASELINE_MEAN.
)


def _make_capa(saving):
    return CAPA(segment_saving=PenalisedScore(saving))


# ---------------------------------------------------------------------------
# Baseline attribute tests
# ---------------------------------------------------------------------------


def test_gaussian_saving_stores_scalar_baseline():
    """Scalar baseline params are broadcast and stored as 1-D arrays after fit."""
    scorer = GaussianSaving(baseline_mean=BASELINE_MEAN, baseline_var=BASELINE_VAR)
    X = make_no_change_X(scorer, loc=BASELINE_MEAN, scale=BASELINE_VAR**0.5)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [BASELINE_MEAN])
    np.testing.assert_array_equal(scorer.baseline_var_, [BASELINE_VAR])


def test_gaussian_saving_broadcasts_to_n_features():
    """A scalar baseline is broadcast to all n_features after fit."""
    scorer = GaussianSaving(baseline_mean=3.0, baseline_var=4.0)
    X = make_no_change_X(scorer, n_features=3, loc=3.0, scale=2.0)
    scorer.fit(X)
    np.testing.assert_array_equal(scorer.baseline_mean_, [3.0, 3.0, 3.0])
    np.testing.assert_array_equal(scorer.baseline_var_, [4.0, 4.0, 4.0])


def test_gaussian_saving_non_positive_var_raises():
    """A non-positive baseline_var must raise ValueError on fit."""
    scorer = GaussianSaving(baseline_mean=0.0, baseline_var=-1.0)
    X = make_no_change_X(scorer, loc=0.0, scale=1.0)
    with pytest.raises(ValueError):
        scorer.fit(X)


# ---------------------------------------------------------------------------
# CAPA sanity tests with valued baselines
# ---------------------------------------------------------------------------


def test_capa_gaussian_saving_finds_no_changepoint():
    """CAPA with a matched Gaussian valued baseline finds no changepoints."""
    scorer = GaussianSaving(baseline_mean=BASELINE_MEAN, baseline_var=BASELINE_VAR)
    X = make_no_change_X(scorer, loc=BASELINE_MEAN, scale=BASELINE_VAR**0.5)
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 0, f"Expected 0 changepoints, got {len(cpts)}: {cpts}"


def test_capa_gaussian_saving_finds_single_changepoint():
    """CAPA with a matched Gaussian valued baseline detects the true changepoint."""
    scorer = GaussianSaving(baseline_mean=BASELINE_MEAN, baseline_var=BASELINE_VAR)
    X = make_single_change_X(
        scorer,
        loc_before=BASELINE_MEAN,
        loc_after=LOC_AFTER,
        scale=BASELINE_VAR**0.5,
    )
    capa = _make_capa(scorer)
    capa.fit(X)
    cpts = capa.predict_changepoints(X)
    assert len(cpts) == 1, f"Expected 1 changepoint, got {len(cpts)}: {cpts}"
    assert (
        abs(cpts[0] - CHANGEPOINT) <= 6
    ), f"Changepoint {cpts[0]} is too far from true changepoint {CHANGEPOINT}."
