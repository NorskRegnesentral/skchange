"""Change detection algorithms."""

from skchange.change_detectors.base import BaseChangeDetector
from skchange.change_detectors.moving_window import MovingWindow
from skchange.change_detectors.pelt import PELT
from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation

BASE_CHANGE_DETECTORS = [BaseChangeDetector]
CHANGE_DETECTORS = [MovingWindow, PELT, SeededBinarySegmentation]

__all__ = BASE_CHANGE_DETECTORS + CHANGE_DETECTORS
