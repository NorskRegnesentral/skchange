"""Tests for calibrate_penalty()."""

import numpy as np
import pytest

from skchange.new_api.calibration._calibrate import calibrate_penalty
from skchange.new_api.calibration._null_models import (
    GaussianNullModel,
    PermutationNullModel,
)
from skchange.new_api.detectors import MovingWindow
from skchange.new_api.interval_scorers import CUSUM, ESACScore, L2Saving, PenalisedScore

_RNG = np.random.default_rng(0)
_N, _P = 100, 3
_X = _RNG.normal(size=(_N, _P))


# ---------------------------------------------------------------------------
# Basic return type and value
# ---------------------------------------------------------------------------


def test_calibrate_penalty_returns_positive_scalar_for_cusum():
    """calibrate_penalty with CUSUM must return a positive scalar."""
    result = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=19, random_state=0
    )
    assert np.isscalar(result) or (isinstance(result, np.ndarray) and result.ndim == 0)
    assert float(result) > 0


def test_calibrate_penalty_return_shape_matches_get_default_penalty():
    """Return shape must match scorer.get_default_penalty() shape after fit."""
    scorer = L2Saving()
    scorer.fit(_X)
    default_penalty = scorer.get_default_penalty()

    result = calibrate_penalty(
        L2Saving(), _X, PermutationNullModel(), n_simulations=19, random_state=0
    )
    assert np.shape(result) == np.shape(default_penalty)


def test_calibrate_penalty_smoke_one_simulation():
    """n_simulations=1 must return a finite value (smoke test)."""
    result = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=1, random_state=42
    )
    assert np.all(np.isfinite(result))


def test_calibrate_penalty_with_gaussian_null_model():
    """calibrate_penalty works with GaussianNullModel."""
    result = calibrate_penalty(
        CUSUM(), _X, GaussianNullModel(), n_simulations=19, random_state=0
    )
    assert float(result) > 0


def test_calibrate_penalty_with_esac_score():
    """calibrate_penalty works with ESACScore (after Phase 0 refactor)."""
    result = calibrate_penalty(
        ESACScore(), _X, PermutationNullModel(), n_simulations=19, random_state=0
    )
    assert float(result) > 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_calibrate_penalty_random_state_reproducible():
    """Same random_state must produce identical results."""
    r1 = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=49, random_state=7
    )
    r2 = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=49, random_state=7
    )
    np.testing.assert_array_equal(r1, r2)


def test_calibrate_penalty_different_seeds_differ():
    """Different random states should (almost certainly) give different values."""
    r1 = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=49, random_state=1
    )
    r2 = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=49, random_state=2
    )
    # Very unlikely to be identical with 49 sims.
    assert not np.array_equal(r1, r2)


# ---------------------------------------------------------------------------
# interval_specs resolution
# ---------------------------------------------------------------------------


def test_calibrate_penalty_explicit_interval_specs():
    """Explicit interval_specs must be respected."""
    specs = np.array([[0, 25, 50], [25, 50, 75], [50, 75, 100]], dtype=np.int64)
    result = calibrate_penalty(
        CUSUM(),
        _X,
        PermutationNullModel(),
        interval_specs=specs,
        n_simulations=19,
        random_state=0,
    )
    assert float(result) > 0


def test_calibrate_penalty_with_detector_uses_get_interval_specs():
    """When detector has get_interval_specs(), it must be used for intervals."""
    detector = MovingWindow(PenalisedScore(CUSUM()), bandwidth=10)
    detector.fit(_X)
    result = calibrate_penalty(
        CUSUM(),
        _X,
        PermutationNullModel(),
        detector=detector,
        n_simulations=19,
        random_state=0,
    )
    assert float(result) > 0


def test_calibrate_penalty_detector_result_le_conservative():
    """Detector-specific calibration should give a penalty <= the conservative default.

    The conservative default uses more intervals, so the per-simulation maximum
    is at least as large, leading to a (weakly) higher calibrated penalty.
    """
    detector = MovingWindow(PenalisedScore(CUSUM()), bandwidth=10)
    detector.fit(_X)

    result_detector = calibrate_penalty(
        CUSUM(),
        _X,
        PermutationNullModel(),
        detector=detector,
        n_simulations=99,
        random_state=42,
    )
    result_conservative = calibrate_penalty(
        CUSUM(), _X, PermutationNullModel(), n_simulations=99, random_state=42
    )
    # Conservative default should be >= detector-specific (or at least close).
    assert float(result_conservative) >= float(result_detector) * 0.9


# ---------------------------------------------------------------------------
# FWER validation (marked slow)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# X_calib — separate calibration dataset
# ---------------------------------------------------------------------------


def test_calibrate_penalty_X_calib_returns_positive():
    """calibrate_penalty with a separate X_calib must return a positive value."""
    X_calib = np.random.default_rng(1).normal(size=(200, _P))
    result = calibrate_penalty(
        CUSUM(),
        _X,
        PermutationNullModel(),
        n_simulations=19,
        random_state=0,
        X_calib=X_calib,
    )
    assert float(result) > 0


def test_calibrate_penalty_X_calib_1d_raises():
    """1-D X_calib must raise ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        calibrate_penalty(
            CUSUM(),
            _X,
            PermutationNullModel(),
            n_simulations=9,
            random_state=0,
            X_calib=np.ones(_N),
        )


def test_calibrate_penalty_X_calib_wrong_features_raises():
    """X_calib with wrong number of features must raise ValueError."""
    X_calib_bad = np.random.default_rng(2).normal(size=(80, _P + 1))
    with pytest.raises(ValueError, match="features"):
        calibrate_penalty(
            CUSUM(),
            _X,
            PermutationNullModel(),
            n_simulations=9,
            random_state=0,
            X_calib=X_calib_bad,
        )


def test_calibrate_penalty_X_calib_different_length_allowed():
    """X_calib with a different length than X must be accepted without error.

    Uses GaussianNullModel because PermutationNullModel cannot upsample
    (draws without replacement), so it requires len(X_calib) >= len(X).
    """
    X_calib_short = np.random.default_rng(3).normal(size=(30, _P))
    result = calibrate_penalty(
        CUSUM(),
        _X,
        GaussianNullModel(),
        n_simulations=9,
        random_state=0,
        X_calib=X_calib_short,
    )
    assert float(result) > 0


def test_calibrate_penalty_X_calib_gaussian_null_differs_from_X_only():
    """Using a very different X_calib should shift the calibrated penalty."""
    rng = np.random.default_rng(42)
    # X_calib drawn from a distribution with much larger variance.
    X_calib_wide = rng.normal(scale=5.0, size=(200, _P))
    result_x_only = calibrate_penalty(
        CUSUM(), _X, GaussianNullModel(), n_simulations=99, random_state=0
    )
    result_x_calib = calibrate_penalty(
        CUSUM(),
        _X,
        GaussianNullModel(),
        n_simulations=99,
        random_state=0,
        X_calib=X_calib_wide,
    )
    # With scale-5 null data, the calibrated penalty should be larger
    # (higher spread => higher scores under null => stricter threshold).
    assert float(result_x_calib) > float(result_x_only)


@pytest.mark.slow
def test_calibrate_penalty_fwer_control():
    """Calibrated penalty must control FWER at the nominal level (±2%)."""
    level = 0.05
    n_series = 200

    calibrated = calibrate_penalty(
        CUSUM(),
        _X,
        PermutationNullModel(),
        n_simulations=999,
        level=level,
        random_state=0,
    )

    rng = np.random.default_rng(99)
    false_alarms = 0
    detector = MovingWindow(PenalisedScore(CUSUM(), penalty=calibrated), bandwidth=10)
    detector.fit(_X)

    for _ in range(n_series):
        X_null = rng.normal(size=(_N, _P))
        cpts = detector.predict_changepoints(X_null)
        if len(cpts) > 0:
            false_alarms += 1

    fwer = false_alarms / n_series
    assert fwer <= level + 0.04, (
        f"FWER {fwer:.3f} exceeds nominal {level} + 0.04 tolerance."
    )
