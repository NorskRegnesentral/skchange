"""Tests for CalibratedDetector."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.calibration._calibrated_detector import CalibratedDetector
from skchange.new_api.detectors import CAPA, PELT, MovingWindow
from skchange.new_api.interval_scorers import CUSUM

_RNG = np.random.default_rng(0)
_N, _P = 100, 2
_X = _RNG.normal(size=(_N, _P))


# ---------------------------------------------------------------------------
# Basic fit and predict
# ---------------------------------------------------------------------------


def test_calibrated_detector_fit_returns_self():
    """fit() must return self."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        resampling="permutation",
        n_simulations=9,
        random_state=0,
    )
    assert cd.fit(_X) is cd


def test_calibrated_detector_fit_stores_detector_():
    """After fit(), detector_ attribute must be set."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert hasattr(cd, "detector_")


def test_calibrated_detector_penalty_scales_nonempty():
    """penalty_scales_ must be a non-empty dict after fit."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert isinstance(cd.penalty_scales_, dict)
    assert len(cd.penalty_scales_) > 0


def test_calibrated_detector_penalty_scale_key_for_moving_window():
    """penalty_scales_ for MovingWindow must have key 'penalty_scale'."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert "penalty_scale" in cd.penalty_scales_
    assert cd.penalty_scales_["penalty_scale"] > 0


def test_calibrated_detector_n_simulations_done():
    """n_simulations_done_ must equal n_simulations after fit."""
    n_sim = 9
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=n_sim,
        random_state=0,
    )
    cd.fit(_X)
    assert cd.n_simulations_done_ == n_sim


def test_calibrated_detector_predict_changepoints_returns_array():
    """predict_changepoints must return a numpy array."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    result = cd.predict_changepoints(_X)
    assert isinstance(result, np.ndarray)


def test_calibrated_detector_predict_before_fit_raises():
    """predict_changepoints before fit must raise NotFittedError."""
    cd = CalibratedDetector(MovingWindow(CUSUM()))
    with pytest.raises(NotFittedError):
        cd.predict_changepoints(_X)


# ---------------------------------------------------------------------------
# CAPA: two penalty_scale params -> two calibrated scales
# ---------------------------------------------------------------------------


def test_calibrated_detector_capa_both_penalty_scales_calibrated():
    """CAPA with include_point_anomalies=True should calibrate both penalty scales."""
    cd = CalibratedDetector(
        CAPA(include_point_anomalies=True), n_simulations=9, random_state=0
    )
    cd.fit(_X)
    assert len(cd.penalty_scales_) == 2
    assert "segment_penalty_scale" in cd.penalty_scales_
    assert "point_penalty_scale" in cd.penalty_scales_


def test_calibrated_detector_capa_segment_only_when_no_point_anomalies():
    """CAPA with include_point_anomalies=False (default) calibrates only segment_penalty_scale."""
    cd = CalibratedDetector(CAPA(), n_simulations=9, random_state=0)
    cd.fit(_X)
    assert len(cd.penalty_scales_) == 1
    assert "segment_penalty_scale" in cd.penalty_scales_
    assert "point_penalty_scale" not in cd.penalty_scales_


# ---------------------------------------------------------------------------
# Gaussian sampler alias
# ---------------------------------------------------------------------------


def test_calibrated_detector_gaussian_resampling():
    """resampling='gaussian' must succeed (parametric sampler, no fit needed)."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        resampling="gaussian",
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert cd.penalty_scales_["penalty_scale"] > 0


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------


def test_calibrated_detector_clone_is_unfitted():
    """clone() of CalibratedDetector must produce an unfitted estimator."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    cloned = clone(cd)
    assert not hasattr(cloned, "detector_")


def test_calibrated_detector_get_params_set_params_roundtrip():
    """get_params / set_params must round-trip correctly."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    params = cd.get_params(deep=False)
    cd.set_params(**params)
    assert cd.get_params(deep=False) == params


# ---------------------------------------------------------------------------
# PELT must raise NotImplementedError
# ---------------------------------------------------------------------------


def test_calibrated_detector_raises_for_pelt():
    """CalibratedDetector with PELT must raise NotImplementedError."""
    cd = CalibratedDetector(PELT(), n_simulations=9, random_state=0)
    with pytest.raises(NotImplementedError):
        cd.fit(_X)


def test_calibrated_detector_raises_for_any_detector_without_penalty_scale():
    """CalibratedDetector raises for any detector with zero penalty_scale params."""
    cd = CalibratedDetector(PELT())
    with pytest.raises(NotImplementedError):
        cd.fit(_X)


# ---------------------------------------------------------------------------
# Default resampling
# ---------------------------------------------------------------------------


def test_calibrated_detector_default_resampling_is_permutation():
    """When resampling is default ('permutation'), fit must succeed."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert hasattr(cd, "detector_")


# ---------------------------------------------------------------------------
# X_calib — separate calibration dataset
# ---------------------------------------------------------------------------

_X_CALIB = np.random.default_rng(10).normal(size=(150, _P))


def test_calibrated_detector_fit_with_X_calib_returns_self():
    """fit(X, X_calib=...) must return self."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    assert cd.fit(_X, X_calib=_X_CALIB) is cd


def test_calibrated_detector_fit_with_X_calib_sets_detector_():
    """fit(X, X_calib=...) must set detector_ attribute."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X, X_calib=_X_CALIB)
    assert hasattr(cd, "detector_")


def test_calibrated_detector_X_calib_1d_raises():
    """1-D X_calib must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    with pytest.raises(ValueError, match="2-D"):
        cd.fit(_X, X_calib=np.ones(_N))


def test_calibrated_detector_X_calib_wrong_features_raises():
    """X_calib with wrong n_features must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    X_calib_bad = np.random.default_rng(5).normal(size=(80, _P + 1))
    with pytest.raises(ValueError, match="features"):
        cd.fit(_X, X_calib=X_calib_bad)


def test_calibrated_detector_X_calib_different_length_allowed():
    """X_calib of a different length than X must be accepted.

    Uses 'gaussian' resampling because PermutationSampler cannot upsample
    (draws without replacement), so it requires len(X_calib) >= len(X).
    """
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        resampling="gaussian",
        n_simulations=9,
        random_state=0,
    )
    X_calib_short = np.random.default_rng(6).normal(size=(30, _P))
    cd.fit(_X, X_calib=X_calib_short)
    assert hasattr(cd, "detector_")


# ---------------------------------------------------------------------------
# X_train — separate scorer / detector training dataset
# ---------------------------------------------------------------------------

_N_TRAIN = 500
_X_TRAIN = np.random.default_rng(20).normal(size=(_N_TRAIN, _P))


def test_calibrated_detector_fit_with_X_train_returns_self():
    """fit(X, X_train=...) must return self."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    assert cd.fit(_X, X_train=_X_TRAIN) is cd


def test_calibrated_detector_fit_with_X_train_sets_detector_():
    """fit(X, X_train=...) must set detector_ attribute."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X, X_train=_X_TRAIN)
    assert hasattr(cd, "detector_")


def test_calibrated_detector_X_train_1d_raises():
    """1-D X_train must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    with pytest.raises(ValueError, match="2-D"):
        cd.fit(_X, X_train=np.ones(_N))


def test_calibrated_detector_X_train_wrong_features_raises():
    """X_train with wrong n_features must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        n_simulations=9,
        random_state=0,
    )
    X_train_bad = np.random.default_rng(5).normal(size=(_N_TRAIN, _P + 1))
    with pytest.raises(ValueError, match="features"):
        cd.fit(_X, X_train=X_train_bad)


def test_calibrated_detector_X_train_backward_compat():
    """X_train=None must give identical results to not passing X_train."""
    cd1 = CalibratedDetector(MovingWindow(CUSUM()), n_simulations=19, random_state=42)
    cd2 = CalibratedDetector(MovingWindow(CUSUM()), n_simulations=19, random_state=42)
    cd1.fit(_X)
    cd2.fit(_X, X_train=None)
    assert cd1.penalty_scales_ == cd2.penalty_scales_


def test_calibrated_detector_X_train_uses_permutation_on_train_data():
    """With X_train, PermutationSampler should sample from X_train rows.

    len(X) < len(X_train), so without X_train the permutation sampler would
    draw from the 100-row X. With X_train, it draws from 500-row X_train.
    Both should succeed and produce a valid positive penalty scale.
    """
    cd = CalibratedDetector(
        MovingWindow(CUSUM()),
        resampling="permutation",
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X, X_train=_X_TRAIN)
    assert cd.penalty_scales_["penalty_scale"] > 0


# ---------------------------------------------------------------------------
# PermutationSampler — replace=False guard
# ---------------------------------------------------------------------------


def test_permutation_no_replace_too_few_training_raises():
    """PermutationSampler(replace=False) must raise ValueError when n_samples > n_train."""
    from skchange.new_api.calibration._null_models import PermutationSampler

    rng = np.random.default_rng(0)
    sampler = PermutationSampler(replace=False).fit(
        np.random.default_rng(1).normal(size=(10, 2))
    )
    with pytest.raises(ValueError, match="replace=False"):
        sampler.sample(50, rng)
