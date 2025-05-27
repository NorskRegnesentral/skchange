"""Change detection algorithms."""

from ._crops import CROPS_PELT
from ._moving_window import MovingWindow
from ._pelt import PELT, JumpPELT
from ._seeded_binseg import SeededBinarySegmentation
from .base import BaseChangeDetector

BASE_CHANGE_DETECTORS = [BaseChangeDetector]
CHANGE_DETECTORS = [CROPS_PELT, MovingWindow, PELT, JumpPELT, SeededBinarySegmentation]

__all__ = BASE_CHANGE_DETECTORS + CHANGE_DETECTORS
