"""Tests for the _calibration_strategy class attribute on detectors."""

import pytest

from skchange.new_api.detectors import (
    CAPA,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.detectors._base import BaseChangeDetector


def test_base_detector_default_is_detection_count():
    assert BaseChangeDetector._calibration_strategy == "detection_count"


@pytest.mark.parametrize(
    "detector_cls",
    [SeededBinarySegmentation, MovingWindow, CircularBinarySegmentation],
    ids=lambda c: c.__name__,
)
def test_scanner_strategy_is_max_score(detector_cls):
    assert detector_cls._calibration_strategy == "max_score"


def test_capa_strategy_is_detection_count():
    # CAPA has no single scalar penalty base (it carries separate segment and
    # point penalties), so ``_discover_knob`` returns ``base=None`` and the
    # closed-form ``max_score`` path is never taken -- calibration always falls
    # through to ``detection_count``. The class tag must say so honestly.
    assert CAPA._calibration_strategy == "detection_count"


def test_pelt_strategy_is_path_search():
    # PELT jointly optimises over all changepoint sets: the single-split score
    # underestimates the true critical penalty, and bisection is slow and
    # approximate. The exact secant path-search computes β* = max_k G_k/k in
    # ~3-6 fits instead of ~15-25.
    assert PELT._calibration_strategy == "path_search"


def test_strategy_accessible_via_getattr_with_default():
    """getattr fallback must also return 'detection_count' for unknown detectors."""

    class UnknownDetector(BaseChangeDetector):
        def fit(self, X, y=None):
            return self

        def predict_changepoints(self, X):
            import numpy as np

            return np.empty(0, dtype=int)

    det = UnknownDetector()
    assert getattr(det, "_calibration_strategy", "detection_count") == "detection_count"


def test_strategy_is_class_level_not_instance_level():
    """The attribute must be on the class, not set per-instance."""
    sbs = SeededBinarySegmentation()
    # Accessing via the class and the instance must agree.
    assert sbs._calibration_strategy == SeededBinarySegmentation._calibration_strategy
