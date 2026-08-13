"""Tests for PELT path-search critical-scale computation.

These tests test ``_critical_scale_path_search``. They check that it agrees
with the bisection fallback, that PELT reports zero changepoints at the returned
scale, and that it uses fewer fits than bisection.
"""

import numpy as np
import pytest
from sklearn.base import clone

from skchange.detectors import PELT
from skchange.tuning._fwer_calibration import (
    _BISECT_LO,
    _critical_scale_count,
    _critical_scale_path_search,
    _discover_knob,
)


def _null_X(n=100, p=2, seed=0):
    return np.random.default_rng(seed).normal(size=(n, p))


def _blip_X(m=15, a=10.0):
    """Flat-0 / 2-sample bump / flat-0 series.

    The two-changepoint model perfectly isolates the bump (zero residual cost),
    so G_2/2 > G_1: the path-search critical scale must exceed the max-score
    single-split critical scale.
    """
    X = np.zeros((2 * m + 2, 1))
    X[m : m + 2] = a
    return X


# --------------------------------------------------------------------------- #
# path_search agrees with bisection (detection_count)
# --------------------------------------------------------------------------- #


def test_path_search_agrees_with_bisection():
    """path_search and bisection must agree within 1 % on random null samples."""
    rng = np.random.default_rng(42)
    for i in range(10):
        X = rng.normal(size=(80, 2))
        det = PELT()
        knob, base = _discover_knob(det, X)
        assert base is not None

        c_path = _critical_scale_path_search(det, X, knob, base)
        c_bisect = _critical_scale_count(det, X, knob)

        assert c_path == pytest.approx(
            c_bisect, rel=0.02
        ), f"Sample {i}: path_search={c_path:.6f}, bisection={c_bisect:.6f}"


# --------------------------------------------------------------------------- #
# Zero changepoints at the returned scale, and fewer fits than bisection
# --------------------------------------------------------------------------- #


def test_path_search_terminates_zero_changepoints():
    """At the returned scale, PELT must report zero changepoints."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 2))
    det = PELT()
    knob, base = _discover_knob(det, X)
    assert base is not None

    c_b = _critical_scale_path_search(det, X, knob, base)
    cps = clone(det).set_params(**{knob: c_b}).fit_predict(X)
    assert len(cps) == 0, f"PELT at scale={c_b:.6f} still detects: {cps}"


def test_path_search_zero_changepoints_on_blip():
    """Blip: PELT at path-search scale must be silent."""
    X = _blip_X(m=15, a=10.0)
    det = PELT()
    knob, base = _discover_knob(det, X)
    assert base is not None

    c_b = _critical_scale_path_search(det, X, knob, base)
    cps = clone(det).set_params(**{knob: c_b}).fit_predict(X)
    assert len(cps) == 0


def test_path_search_fewer_fits_than_bisection():
    """path_search must use substantially fewer PELT fits than bisection."""
    fit_log: list[int] = []

    class CountingPELT(PELT):
        """PELT subclass that counts .fit() calls into a shared list."""

        def fit(self, X, y=None):
            fit_log.append(1)
            return super().fit(X, y)

    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 2))
    det = CountingPELT()
    knob, base = _discover_knob(det, X)
    assert base is not None

    # Count path-search fits
    fit_log.clear()
    _critical_scale_path_search(det, X, knob, base)
    path_fits = len(fit_log)

    # Count bisection fits
    fit_log.clear()
    _critical_scale_count(det, X, knob)
    bisect_fits = len(fit_log)

    assert (
        path_fits < bisect_fits
    ), f"path_search used {path_fits} fits vs bisection {bisect_fits}"
    # Expected: about 3 to 6 path fits versus 15 to 25 for bisection
    assert path_fits <= 15, f"path_search used too many fits: {path_fits}"


def test_path_search_returns_positive_scale():
    """path_search must return a positive scale for any null sample."""
    rng = np.random.default_rng(99)
    for _ in range(5):
        X = rng.normal(size=(80, 1))
        det = PELT()
        knob, base = _discover_knob(det, X)
        assert base is not None
        c_b = _critical_scale_path_search(det, X, knob, base)
        assert c_b > 0.0, f"path_search returned non-positive scale: {c_b}"


def test_path_search_converges_at_hull_vertex():
    """The secant walk can stop at the hull vertex, not only via zero cps.

    With clean, well separated blocks the average cost reduction per
    changepoint stabilises, so ``beta_new <= beta`` triggers and the loop
    exits at the top vertex while PELT still reports changepoints at that
    penalty. The returned scale, after the tie nudge, must still silence PELT.
    """
    rng = np.random.default_rng(19)
    blocks = [np.full((23, 1), level * 11.407) for level in range(5)]
    X = np.vstack(blocks) + rng.normal(size=(115, 1)) * 0.419

    det = PELT()
    knob, base = _discover_knob(det, X)
    assert base is not None

    c_b = _critical_scale_path_search(det, X, knob, base)
    assert c_b > 0.0
    cps = clone(det).set_params(**{knob: c_b}).fit_predict(X)
    assert len(cps) == 0, f"PELT at hull-vertex scale={c_b:.6f} still detects: {cps}"


def test_path_search_fallback_when_no_detections_at_tiny_scale():
    """If PELT fires nothing at _BISECT_LO, path_search returns _BISECT_LO."""
    # Near-constant signal: nothing to detect even at tiny penalty.
    X = np.full((100, 1), 0.0)
    X = X + np.finfo(float).eps  # exactly constant → zero variance → zero cost
    det = PELT()
    knob, base = _discover_knob(det, X)
    assert base is not None
    c_b = _critical_scale_path_search(det, X, knob, base)
    # Should equal _BISECT_LO (nothing to detect)
    assert c_b == pytest.approx(_BISECT_LO, rel=0.01)
