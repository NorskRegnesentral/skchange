"""Change detection algorithms."""

from skchange._deprecation import warn_old_api

warn_old_api()

from ._crops import CROPS
from ._moving_window import MovingWindow
from ._pelt import PELT
from ._seeded_binseg import SeededBinarySegmentation
from .base import BaseChangeDetector

BASE_CHANGE_DETECTORS = [BaseChangeDetector]
CHANGE_DETECTORS = [CROPS, MovingWindow, PELT, SeededBinarySegmentation]

__all__ = BASE_CHANGE_DETECTORS + CHANGE_DETECTORS
