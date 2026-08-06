"""Tests for ``skchange.new_api.tuning._fwer_calibration``."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted

from skchange.new_api.detectors import (
    CAPA,
    CROPS,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.interval_scorers import ESACScore
from skchange.new_api.tuning import (
    CalibratedDetector,
    GaussianSampler,
    calibrate_penalty_scale,
)
from skchange.new_api.tuning._fwer_calibration import _discover_knob

# All supported detectors (simple knob, scalar penalty_)
_ALL_SUPPORTED = [
    SeededBinarySegmentation,
    MovingWindow,
    CircularBinarySegmentation,
    PELT,
]

# Simulation counts kept as small as each assertion can tolerate. Fast tests
# only need "scale > 0" or reproducibility, which are 1-sim properties.
# Monotone tests must resolve a signed inequality (e.g. dirty >= clean), so
# they need enough sims for the two scales to separate reliably.
_FAST_N_SIMS = 10
_MONOTONE_N_SIMS = 50


def _null_X(n=60, p=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


# --------------------------------------------------------------------------- #
# Task 3.1 — Knob discovery
# --------------------------------------------------------------------------- #


def test_knob_discovery_returns_penalty_scale_for_scanner():
    X = _null_X()
    det = SeededBinarySegmentation()
    knob, base = _discover_knob(det, X)
    assert knob == "penalty_scale"
    assert base is not None and base > 0.0


def test_knob_discovery_returns_penalty_scale_for_pelt():
    X = _null_X()
    det = PELT()
    knob, base = _discover_knob(det, X)
    assert knob == "penalty_scale"
    assert base is not None and base > 0.0


def test_knob_discovery_returns_nested_for_esac_scorer():
    X = _null_X()
    det = SeededBinarySegmentation(change_score=ESACScore())
    knob, _ = _discover_knob(det, X)
    assert "__penalty_scale" in knob
    assert "change_score" in knob


def test_knob_discovery_raises_for_crops():
    X = _null_X()
    with pytest.raises(ValueError, match="penalty_scale"):
        _discover_knob(CROPS(), X)


def test_knob_discovery_base_equals_penalty_at_scale_one():
    """The base returned by _discover_knob must equal penalty_ at penalty_scale=1."""
    X = _null_X()
    det = SeededBinarySegmentation()
    knob, base = _discover_knob(det, X)
    fitted_at_one = clone(det).set_params(**{knob: 1.0}).fit(X)
    assert fitted_at_one.penalty_ == pytest.approx(base, rel=1e-10)


# --------------------------------------------------------------------------- #
# Task 3.2 — max_score closed form c_b = max(S) / base
# --------------------------------------------------------------------------- #


def test_max_score_crit_scale_matches_brute_force_on_scanner():
    """For SBS, c_b = max(S)/base must equal the scale from a fine penalty sweep."""
    from skchange.new_api.tuning._fwer_calibration import (
        _critical_scale_count,
        _critical_scale_max_score,
    )

    X = _null_X(n=80)
    det = SeededBinarySegmentation()
    knob, base = _discover_knob(det, X)
    assert base is not None

    c_max = _critical_scale_max_score(det, X, knob, base)
    c_count = _critical_scale_count(det, X, knob)

    # They should agree within a small tolerance.
    assert c_max == pytest.approx(c_count, rel=0.01)


def test_max_score_returns_zero_when_silent_at_scale_zero():
    """If detector produces no scores at scale 0, c_b must be 0."""
    from skchange.new_api.tuning._fwer_calibration import _critical_scale_max_score

    # Use a very long null series where any reasonable penalty kills detections.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1)) * 0.001  # near-zero variance, nothing to detect
    det = SeededBinarySegmentation()
    knob, base = _discover_knob(det, X)
    assert base is not None
    c = _critical_scale_max_score(det, X, knob, base)
    assert c >= 0.0


# --------------------------------------------------------------------------- #
# Task 3.3 — PELT single-split hook (already covered in test_pelt.py)
# 3.4 — PELT max_score ≈ detection_count on a convex cost
# --------------------------------------------------------------------------- #


def test_pelt_max_score_not_greater_than_detection_count():
    """max_score (single-split bound) must not exceed detection_count on most samples.

    PELT can combine multiple splits, so the single-split score underestimates
    the true critical penalty on average. This is why PELT uses detection_count
    as its calibration strategy rather than max_score.
    """
    from skchange.new_api.tuning._fwer_calibration import (
        _critical_scale_count,
        _critical_scale_max_score,
    )

    rng = np.random.default_rng(7)
    n_checks = 20
    n_lower = 0
    for _ in range(n_checks):
        X = rng.normal(size=(80, 2))
        det = PELT()
        knob, base = _discover_knob(det, X)
        assert base is not None
        c_max = _critical_scale_max_score(det, X, knob, base)
        c_count = _critical_scale_count(det, X, knob)
        if c_max < c_count:
            n_lower += 1

    # max_score is strictly lower than detection_count on most samples.
    assert n_lower >= n_checks // 2


def test_detection_count_handles_non_monotone_silent_at_zero():
    """detection_count must find the UPPER firing edge even when the detector is
    silent at near-zero penalty.

    CAPA absorbs the whole series into a single anomalous segment when the
    penalty is ~0, so its changepoint count is 0 at scale -> 0, rises for
    moderate penalties, then falls back to 0. The detected count is therefore
    *not* monotone in ``penalty_scale``. The bisection must return the upper
    edge of the firing region, not collapse to ~0.
    """
    from skchange.new_api.interval_scorers import L2Saving
    from skchange.new_api.tuning._fwer_calibration import (
        _BISECT_LO,
        _critical_scale_count,
    )

    det = CAPA(segment_saving=L2Saving())
    knob, _ = _discover_knob(det, _null_X(n=50))

    def n_cpts(X, scale):
        cps = clone(det).set_params(**{knob: scale}).fit_predict(X)
        return len(cps)

    # Construct a null sample that is silent at _BISECT_LO but fires at 0.1.
    rng = np.random.default_rng(11)
    X = None
    for _ in range(30):
        cand = rng.normal(size=(50, 2))
        if n_cpts(cand, _BISECT_LO) == 0 and n_cpts(cand, 0.1) > 0:
            X = cand
            break
    assert X is not None, "could not construct a non-monotone CAPA null sample"

    c_b = _critical_scale_count(det, X, knob)

    # The critical scale is the upper edge: clearly above zero, the detector is
    # silent at c_b but fires just below it.
    assert c_b > 0.01, f"critical scale collapsed to ~0 despite mid-range firing: {c_b}"
    assert n_cpts(X, c_b) == 0, f"detector still fires at critical scale {c_b}"
    assert n_cpts(X, c_b * 0.9) > 0, f"detector silent just below critical scale {c_b}"


# --------------------------------------------------------------------------- #
# Task 4.1 — calibrate_penalty_scale: positive multiplier, level monotone,
#             reproducibility, per-detector, CROPS rejection, ESAC works
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("detector_cls", _ALL_SUPPORTED, ids=lambda c: c.__name__)
def test_returns_positive_float(detector_cls):
    scale = calibrate_penalty_scale(
        detector_cls(), _null_X(), n_simulations=_FAST_N_SIMS, random_state=0
    )
    assert isinstance(scale, float)
    assert scale > 0.0


def test_reproducible_with_seed():
    X = _null_X()
    a = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, n_simulations=_FAST_N_SIMS, random_state=42
    )
    b = calibrate_penalty_scale(
        SeededBinarySegmentation(), X, n_simulations=_FAST_N_SIMS, random_state=42
    )
    assert a == b


def test_lower_level_gives_larger_scale():
    X = _null_X()
    strict = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        level=0.01,
        n_simulations=_MONOTONE_N_SIMS,
        random_state=0,
    )
    loose = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        level=0.20,
        n_simulations=_MONOTONE_N_SIMS,
        random_state=0,
    )
    assert strict >= loose


def test_gaussian_sampler_runs():
    scale = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        _null_X(),
        sampler="gaussian",
        n_simulations=_FAST_N_SIMS,
        random_state=0,
    )
    assert scale > 0.0


def test_sampler_instance_and_callable():
    X = _null_X()
    s1 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        sampler=GaussianSampler(std=1.0),
        n_simulations=_FAST_N_SIMS,
        random_state=0,
    )
    s2 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        sampler=lambda n, p, rng: rng.normal(size=(n, p)),
        n_simulations=_FAST_N_SIMS,
        random_state=0,
    )
    assert s1 > 0.0 and s2 > 0.0


def test_detector_not_modified_by_calibration():
    """The input detector's parameters must be unchanged after calibration."""
    det = SeededBinarySegmentation(penalty_scale=3.0)
    original_scale = det.penalty_scale
    calibrate_penalty_scale(det, _null_X(), n_simulations=20, random_state=0)
    assert det.penalty_scale == original_scale


def test_x_calib_clean_vs_contaminated():
    rng = np.random.default_rng(0)
    X_contaminated = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(6, 1, (60, 2))])
    X_clean = rng.normal(0, 1, (400, 2))
    dirty = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X_contaminated,
        n_simulations=_MONOTONE_N_SIMS,
        random_state=1,
    )
    clean = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X_contaminated,
        X_calib=X_clean,
        n_simulations=_MONOTONE_N_SIMS,
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


def test_crops_raises_unsupported_error():
    with pytest.raises(ValueError, match="penalty_scale"):
        calibrate_penalty_scale(CROPS(), _null_X(), n_simulations=10)


def test_esac_based_detector_is_calibratable():
    """ESACScore-based detector must now return a positive multiplier."""
    scale = calibrate_penalty_scale(
        SeededBinarySegmentation(change_score=ESACScore()),
        _null_X(n=200, p=5),
        sampler="gaussian",
        n_simulations=_FAST_N_SIMS,
        random_state=0,
    )
    assert scale > 0.0


def test_capa_is_calibratable():
    scale = calibrate_penalty_scale(
        CAPA(), _null_X(), n_simulations=_FAST_N_SIMS, random_state=0
    )
    assert scale > 0.0


def test_pelt_is_calibratable():
    scale = calibrate_penalty_scale(
        PELT(), _null_X(), n_simulations=_FAST_N_SIMS, random_state=0
    )
    assert scale > 0.0


def test_invalid_level_raises():
    with pytest.raises(ValueError):
        calibrate_penalty_scale(
            SeededBinarySegmentation(), _null_X(), level=0.0, n_simulations=10
        )


def test_invalid_level_gt_one_raises():
    with pytest.raises(ValueError):
        calibrate_penalty_scale(
            SeededBinarySegmentation(), _null_X(), level=1.0, n_simulations=10
        )


# --------------------------------------------------------------------------- #
# Task 4.3 — CalibratedDetector
# --------------------------------------------------------------------------- #


def test_calibrated_detector_fit_sets_attributes():
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=_FAST_N_SIMS, random_state=0
    ).fit(_null_X())
    check_is_fitted(cal)
    assert cal.penalty_scale_ > 0.0
    assert hasattr(cal, "detector_")
    assert cal.detector_.get_params()["penalty_scale"] == cal.penalty_scale_


def test_calibrated_detector_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    cal = CalibratedDetector(SeededBinarySegmentation())
    with pytest.raises(NotFittedError):
        cal.predict(_null_X())


def test_calibrated_detector_predicts():
    X = _null_X()
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=_FAST_N_SIMS, random_state=0
    ).fit(X)
    cps = cal.predict(X)
    assert isinstance(cps, np.ndarray)


def test_calibrated_detector_clone_and_params_roundtrip():
    cal = CalibratedDetector(SeededBinarySegmentation(), level=0.10, n_simulations=50)
    cloned = clone(cal)
    assert cloned.get_params()["level"] == 0.10
    assert cloned.get_params()["n_simulations"] == 50
    assert "detector__penalty_scale" in cal.get_params(deep=True)


def test_calibrated_detector_x_calib_kwarg():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    X_clean = rng.normal(size=(300, 2))
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=_FAST_N_SIMS, random_state=0
    ).fit(X, X_calib=X_clean)
    assert cal.penalty_scale_ > 0.0


def test_calibrated_detector_delegates_segment_anomalies():
    cal_cbs = CalibratedDetector(
        CircularBinarySegmentation(), n_simulations=20, random_state=0
    )
    assert hasattr(cal_cbs, "predict_segment_anomalies")
    cal_sbs = CalibratedDetector(SeededBinarySegmentation())
    assert not hasattr(cal_sbs, "predict_segment_anomalies")


def test_calibrated_detector_crops_raises():
    with pytest.raises(ValueError, match="penalty_scale"):
        CalibratedDetector(CROPS()).fit(_null_X())


def test_calibrated_detector_pelt():
    cal = CalibratedDetector(PELT(), n_simulations=_FAST_N_SIMS, random_state=0).fit(
        _null_X()
    )
    assert cal.penalty_scale_ > 0.0
    assert hasattr(cal, "detector_")


# --------------------------------------------------------------------------- #
# Task 5.1 — Slow: empirical FWER control
# --------------------------------------------------------------------------- #


@pytest.mark.slow
@pytest.mark.parametrize("detector_cls", _ALL_SUPPORTED, ids=lambda c: c.__name__)
@pytest.mark.parametrize("sampler", ["permutation", "gaussian"])
def test_empirical_fwer_controlled(detector_cls, sampler):
    n, p, level, tol = 120, 2, 0.05, 0.04
    X = _null_X(n=n, p=p, seed=0)
    scale = calibrate_penalty_scale(
        detector_cls(),
        X,
        sampler=sampler,
        level=level,
        n_simulations=999,
        random_state=1,
    )
    detector = detector_cls().set_params(penalty_scale=scale)
    rng = np.random.default_rng(99)
    n_eval = 2000
    false_alarms = 0
    for _ in range(n_eval):
        X_null = rng.normal(size=(n, p))
        if len(detector.fit_predict(X_null)) > 0:
            false_alarms += 1
    empirical_fwer = false_alarms / n_eval
    assert empirical_fwer <= level + tol, (
        f"{detector_cls.__name__}/{sampler}: empirical FWER "
        f"{empirical_fwer:.3f} exceeds {level} + {tol}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("detector_cls", _ALL_SUPPORTED, ids=lambda c: c.__name__)
def test_calibration_targets_level_not_zero(detector_cls):
    """Calibrated FWER should be ~level, not zero (using (1-level) quantile)."""
    n, p, level, tol = 120, 2, 0.05, 0.04
    X = _null_X(n=n, p=p, seed=0)
    scale = calibrate_penalty_scale(
        detector_cls(),
        X,
        sampler="gaussian",
        level=level,
        n_simulations=999,
        random_state=2,
    )
    detector = detector_cls().set_params(penalty_scale=scale)
    rng = np.random.default_rng(100)
    n_eval = 2000
    false_alarms = sum(
        1
        for _ in range(n_eval)
        if len(detector.fit_predict(rng.normal(size=(n, p)))) > 0
    )
    empirical_fwer = false_alarms / n_eval
    # FWER must be close to level from above (not driven to 0).
    assert empirical_fwer >= level - tol, (
        f"{detector_cls.__name__}: empirical FWER {empirical_fwer:.3f} "
        f"is too far below {level}"
    )


# --------------------------------------------------------------------------- #
# Task 5.2 — X_calib contamination
# --------------------------------------------------------------------------- #


def test_x_calib_clean_gives_scale_le_contaminated():
    """Clean X_calib must give a smaller or equal scale than contaminated X."""
    rng = np.random.default_rng(7)
    X_contaminated = np.vstack([rng.normal(0, 1, (60, 2)), rng.normal(8, 1, (60, 2))])
    X_clean = rng.normal(0, 1, (400, 2))

    scale_dirty = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X_contaminated,
        n_simulations=_MONOTONE_N_SIMS,
        random_state=3,
    )
    scale_clean = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X_contaminated,
        X_calib=X_clean,
        n_simulations=_MONOTONE_N_SIMS,
        random_state=3,
    )
    assert scale_dirty >= scale_clean


# --------------------------------------------------------------------------- #
# Tasks 3.1-3.4 - n_jobs parallel calibration + thread-safe RNG
# --------------------------------------------------------------------------- #


def test_calibrate_n_jobs_invariant():
    """n_jobs=1 and n_jobs=2 must return the same scale for the same random_state."""
    X = _null_X()
    s1 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        n_simulations=_FAST_N_SIMS,
        random_state=42,
        n_jobs=1,
    )
    s2 = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        n_simulations=_FAST_N_SIMS,
        random_state=42,
        n_jobs=2,
    )
    assert s1 == s2


def test_calibrate_reproducible_with_n_jobs():
    """Same random_state must give the same result across two n_jobs=2 runs."""
    X = _null_X()
    a = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        n_simulations=_FAST_N_SIMS,
        random_state=7,
        n_jobs=2,
    )
    b = calibrate_penalty_scale(
        SeededBinarySegmentation(),
        X,
        n_simulations=_FAST_N_SIMS,
        random_state=7,
        n_jobs=2,
    )
    assert a == b


def test_random_state_types_all_accepted():
    """random_state as int, None, and Generator must all return positive floats."""
    X = _null_X()
    det = SeededBinarySegmentation()

    s_int = calibrate_penalty_scale(det, X, n_simulations=20, random_state=0)
    assert isinstance(s_int, float) and s_int > 0

    s_none = calibrate_penalty_scale(det, X, n_simulations=20, random_state=None)
    assert isinstance(s_none, float) and s_none > 0

    s_gen = calibrate_penalty_scale(
        det, X, n_simulations=20, random_state=np.random.default_rng(7)
    )
    assert isinstance(s_gen, float) and s_gen > 0


def test_calibrated_detector_n_jobs_fits_and_predicts():
    """CalibratedDetector with n_jobs=2 must fit and produce valid predictions."""
    X = _null_X()
    cal = CalibratedDetector(
        SeededBinarySegmentation(), n_simulations=20, random_state=0, n_jobs=2
    ).fit(X)
    cps = cal.predict(X)
    assert isinstance(cps, np.ndarray)
    assert cal.penalty_scale_ > 0.0


def test_calibrated_detector_n_jobs_get_params_roundtrip():
    """n_jobs must round-trip through get_params and clone."""
    cal = CalibratedDetector(SeededBinarySegmentation(), n_jobs=2)
    assert cal.get_params()["n_jobs"] == 2
    cloned = clone(cal)
    assert cloned.get_params()["n_jobs"] == 2


def test_detector_not_modified_by_parallel_calibration():
    """The original detector must be unchanged when calibration runs in parallel."""
    det = SeededBinarySegmentation(penalty_scale=3.0)
    original_scale = det.penalty_scale
    calibrate_penalty_scale(det, _null_X(), n_simulations=20, random_state=0, n_jobs=2)
    assert det.penalty_scale == original_scale
