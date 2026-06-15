"""Tests for ``skchange.new_api.tuning._fwer_calibration``."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors import (
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.interval_scorers import ESACScore
from skchange.new_api.tuning import (
    CalibratedDetector,
    GaussianMCSampler,
    calibrate_penalty_scale,
)

# Single-penalty detectors targeted by v1.
_DETECTORS = [
    SeededBinarySegmentation,
    MovingWindow,
    CircularBinarySegmentation,
]


def _null_X(n=120, p=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


# --------------------------------------------------------------------------- #
# Engine: calibrate_penalty_scale
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("detector_cls", _DETECTORS, ids=lambda c: c.__name__)
def test_returns_positive_float(detector_cls):
    scale = calibrate_penalty_scale(
        detector_cls(), _null_X(), n_simulations=99, random_state=0
    )
    assert isinstance(scale, float)
    assert scale > 0.0


def test_reproducible_with_seed():
    X = _null_X()
    a = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, n_simulations=99, random_state=42
    )
    b = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, n_simulations=99, random_state=42
    )
    assert a == b


def test_lower_level_gives_larger_scale():
    X = _null_X()
    strict = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, level=0.01, n_simulations=499, random_state=0
    )
    loose = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, level=0.20, n_simulations=499, random_state=0
    )
    assert strict > loose


def test_gaussian_sampler_runs():
    scale = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        _null_X(),
        sampler="gaussian",
        n_simulations=99,
        random_state=0,
    )
    assert scale > 0.0


def test_sampler_instance_and_callable():
    X = _null_X()
    s1 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        sampler=GaussianMCSampler(std=1.0),
        n_simulations=99,
        random_state=0,
    )
    s2 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        sampler=lambda n, p, rng: rng.normal(size=(n, p)),
        n_simulations=99,
        random_state=0,
    )
    assert s1 > 0.0 and s2 > 0.0


def test_x_calib_clean_vs_contaminated():
    # A contaminated X (real change) inflates null scores -> larger (more
    # conservative) scale than calibrating from clean data.
    rng = np.random.default_rng(0)
    X_contaminated = np.vstack(
        [rng.normal(0, 1, (60, 2)), rng.normal(6, 1, (60, 2))]
    )
    X_clean = rng.normal(0, 1, (400, 2))
    dirty = calibrate_penalty_scale(
        SeededBinarySegmentation(), X_contaminated, n_simulations=299, random_state=1
    )
    clean = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X_contaminated,
        X_calib=X_clean,
        n_simulations=299,
        random_state=1,
    )
    assert dirty > clean


def test_x_calib_feature_mismatch_raises():
    with pytest.raises(ValueError, match="features"):
        calibrate_penalty_scale(
            SeededBinarySegmentation(),
            _null_X(p=2),
            X_calib=_null_X(p=3),
            n_simulations=10,
        )


def test_array_penalty_uses_bisection_path():
    X = _null_X(p=3)
    scale = calibrate_penalty_scale(
        SeededBinarySegmentation(penalty=np.array([4.0, 7.0, 9.0]), agg="sum"),
        X,
        n_simulations=99,
        random_state=0,
    )
    assert scale > 0.0


def test_self_penalised_scorer_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="self-penalised|penalty_ is None"):
        calibrate_penalty_scale(
            SeededBinarySegmentation(change_score=ESACScore()),
            _null_X(),
            n_simulations=10,
        )


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        calibrate_penalty_scale(
            SeededBinarySegmentation(), _null_X(), level=0.0, n_simulations=10
        )


# --------------------------------------------------------------------------- #
# Meta-estimator: CalibratedDetector
# --------------------------------------------------------------------------- #
def test_calibrated_detector_fit_sets_attributes():
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=99, random_state=0
    ).fit(_null_X())
    check_is_fitted(cal)
    assert cal.penalty_scale_ > 0.0
    assert hasattr(cal, "detector_")
    assert cal.detector_.get_params()["penalty_scale"] == cal.penalty_scale_


def test_calibrated_detector_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    cal = CalibratedDetector(SeededBinarySegmentation())
    with pytest.raises(NotFittedError):
        cal.predict_changepoints(_null_X())


def test_calibrated_detector_predicts():
    X = _null_X()
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=99, random_state=0
    ).fit(X)
    cps = cal.predict_changepoints(X)
    assert isinstance(cps, np.ndarray)


def test_calibrated_detector_clone_and_params_roundtrip():
    cal = CalibratedDetector(
        SeededBinarySegmentation(), level=0.10, n_simulations=50
    )
    cloned = clone(cal)
    assert cloned.get_params()["level"] == 0.10
    assert cloned.get_params()["n_simulations"] == 50
    # Nested detector params are exposed.
    assert "detector__penalty_scale" in cal.get_params(deep=True)


def test_calibrated_detector_x_calib_kwarg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    X_clean = rng.normal(size=(300, 2))
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=99, random_state=0
    ).fit(X, X_calib=X_clean)
    assert cal.penalty_scale_ > 0.0


def test_calibrated_detector_delegates_segment_anomalies():
    # CircularBinarySegmentation exposes predict_segment_anomalies; SeededBinSeg
    # does not -> available_if should hide it there.
    cal_cbs = CalibratedDetector(
        CircularBinarySegmentation(), n_simulations=20, random_state=0
    )
    assert hasattr(cal_cbs, "predict_segment_anomalies")
    cal_sbs = CalibratedDetector(SeededBinarySegmentation())
    assert not hasattr(cal_sbs, "predict_segment_anomalies")


# --------------------------------------------------------------------------- #
# Slow: empirical FWER control
# --------------------------------------------------------------------------- #
@pytest.mark.slow
@pytest.mark.parametrize("detector_cls", _DETECTORS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("sampler", ["permutation", "gaussian"])
def test_empirical_fwer_controlled(detector_cls, sampler):
    n, p, level, tol = 120, 2, 0.05, 0.04
    X = _null_X(n=n, p=p, seed=0)
    scale = calibrate_penalty_scale(
        detector_cls(), X, sampler=sampler, level=level,
        n_simulations=999, random_state=1,
    )
    detector = detector_cls().set_params(penalty_scale=scale)
    rng = np.random.default_rng(99)
    n_eval = 2000
    false_alarms = 0
    for _ in range(n_eval):
        X_null = rng.normal(size=(n, p))
        detector.fit(X_null)
        if len(detector.predict_changepoints(X_null)) > 0:
            false_alarms += 1
    empirical_fwer = false_alarms / n_eval
    assert empirical_fwer <= level + tol, (
        f"{detector_cls.__name__}/{sampler}: empirical FWER "
        f"{empirical_fwer:.3f} exceeds {level} + {tol}"
    )
