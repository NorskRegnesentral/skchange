"""Tests for the calibration_strategy tag on detectors."""

import numpy as np
import pytest
from sklearn.utils import get_tags

from skchange.new_api.detectors import (
    CAPA,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.detectors._base import BaseChangeDetector


def _strategy(detector) -> str:
    return get_tags(detector).change_detector_tags.calibration_strategy


class _DummyDetector(BaseChangeDetector):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.empty(0, dtype=int)


def test_base_detector_default_is_detection_count():
    assert _strategy(_DummyDetector()) == "detection_count"


@pytest.mark.parametrize(
    "detector_cls",
    [SeededBinarySegmentation, MovingWindow, CircularBinarySegmentation],
    ids=lambda c: c.__name__,
)
def test_scanner_strategy_is_max_score(detector_cls):
    assert _strategy(detector_cls()) == "max_score"


def test_capa_strategy_is_detection_count():
    assert _strategy(CAPA()) == "detection_count"


def test_pelt_strategy_is_path_search():
    assert _strategy(PELT()) == "path_search"
