"""Tests for CalibratedDetector."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skchange.new_api.calibration._calibrated_detector import CalibratedDetector
from skchange.new_api.calibration._null_models import PermutationNullModel
from skchange.new_api.detectors import CAPA, PELT, MovingWindow
from skchange.new_api.interval_scorers import (
    CUSUM,
    L2Saving,
    PenalisedScore,
)

_RNG = np.random.default_rng(0)
_N, _P = 100, 2
_X = _RNG.normal(size=(_N, _P))


# ---------------------------------------------------------------------------
# Basic fit and predict
# ---------------------------------------------------------------------------


def test_calibrated_detector_fit_returns_self():
    """fit() must return self."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        null_model=PermutationNullModel(),
        n_simulations=9,
        random_state=0,
    )
    assert cd.fit(_X) is cd


def test_calibrated_detector_fit_stores_detector_():
    """After fit(), detector_ attribute must be set."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert hasattr(cd, "detector_")


def test_calibrated_detector_calibrated_penalties_nonempty():
    """calibrated_penalties_ must be a non-empty dict after fit."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    assert isinstance(cd.calibrated_penalties_, dict)
    assert len(cd.calibrated_penalties_) > 0


def test_calibrated_detector_n_simulations_done():
    """n_simulations_done_ must equal n_simulations after fit."""
    n_sim = 9
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=n_sim,
        random_state=0,
    )
    cd.fit(_X)
    assert cd.n_simulations_done_ == n_sim


def test_calibrated_detector_predict_changepoints_returns_array():
    """predict_changepoints must return a numpy array."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    result = cd.predict_changepoints(_X)
    assert isinstance(result, np.ndarray)


def test_calibrated_detector_predict_before_fit_raises():
    """predict_changepoints before fit must raise NotFittedError."""
    cd = CalibratedDetector(MovingWindow(PenalisedScore(CUSUM())))
    with pytest.raises(NotFittedError):
        cd.predict_changepoints(_X)


# ---------------------------------------------------------------------------
# CAPA: two PenalisedScore params -> two calibrated penalties
# ---------------------------------------------------------------------------


def test_calibrated_detector_capa_both_penalties_calibrated():
    """With explicit point_saving, CAPA should have 2 calibrated penalties."""
    segment_saving = PenalisedScore(L2Saving())
    point_saving = PenalisedScore(L2Saving())
    capa = CAPA(segment_saving=segment_saving, point_saving=point_saving)
    cd = CalibratedDetector(capa, n_simulations=9, random_state=0)
    cd.fit(_X)
    assert len(cd.calibrated_penalties_) == 2


# ---------------------------------------------------------------------------
# sklearn contract
# ---------------------------------------------------------------------------


def test_calibrated_detector_clone_is_unfitted():
    """clone() of CalibratedDetector must produce an unfitted estimator."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    cloned = clone(cd)
    assert not hasattr(cloned, "detector_")


def test_calibrated_detector_get_params_set_params_roundtrip():
    """get_params / set_params must round-trip correctly."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
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
    with pytest.raises(NotImplementedError, match="PenalisedScore"):
        cd.fit(_X)


def test_calibrated_detector_raises_for_any_detector_without_penalised_score():
    """CalibratedDetector raises for any detector with zero PenalisedScore params."""
    # PELT is the canonical example; the check is general.
    cd = CalibratedDetector(PELT())
    with pytest.raises(NotImplementedError):
        cd.fit(_X)


# ---------------------------------------------------------------------------
# Default null model
# ---------------------------------------------------------------------------


def test_calibrated_detector_default_null_model_is_permutation():
    """When null_model=None, the default must be PermutationNullModel."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        null_model=None,
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X)
    # Just check that it succeeds (PermutationNullModel is the default).
    assert hasattr(cd, "detector_")


# ---------------------------------------------------------------------------
# X_calib — separate calibration dataset
# ---------------------------------------------------------------------------

_X_CALIB = np.random.default_rng(10).normal(size=(150, _P))


def test_calibrated_detector_fit_with_X_calib_returns_self():
    """fit(X, X_calib=...) must return self."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    assert cd.fit(_X, X_calib=_X_CALIB) is cd


def test_calibrated_detector_fit_with_X_calib_sets_detector_():
    """fit(X, X_calib=...) must set detector_ attribute."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    cd.fit(_X, X_calib=_X_CALIB)
    assert hasattr(cd, "detector_")


def test_calibrated_detector_X_calib_1d_raises():
    """1-D X_calib must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    with pytest.raises(ValueError, match="2-D"):
        cd.fit(_X, X_calib=np.ones(_N))


def test_calibrated_detector_X_calib_wrong_features_raises():
    """X_calib with wrong n_features must raise ValueError."""
    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        n_simulations=9,
        random_state=0,
    )
    X_calib_bad = np.random.default_rng(5).normal(size=(80, _P + 1))
    with pytest.raises(ValueError, match="features"):
        cd.fit(_X, X_calib=X_calib_bad)


def test_calibrated_detector_X_calib_different_length_allowed():
    """X_calib of a different length than X must be accepted.

    Uses GaussianNullModel because PermutationNullModel cannot upsample
    (draws without replacement), so it requires len(X_calib) >= len(X).
    """
    from skchange.new_api.calibration._null_models import GaussianNullModel

    cd = CalibratedDetector(
        MovingWindow(PenalisedScore(CUSUM())),
        null_model=GaussianNullModel(),
        n_simulations=9,
        random_state=0,
    )
    X_calib_short = np.random.default_rng(6).normal(size=(30, _P))
    cd.fit(_X, X_calib=X_calib_short)
    assert hasattr(cd, "detector_")
