"""Change detection algorithms."""

from skchange.change_detectors.base import ChangeDetector
from skchange.change_detectors.moscore import Moscore
from skchange.change_detectors.pelt import Pelt
from skchange.change_detectors.seeded_binseg import SeededBinarySegmentation

BASE_CHANGE_DETECTORS = [ChangeDetector]
CHANGE_DETECTORS = [Moscore, Pelt, SeededBinarySegmentation]

__all__ = BASE_CHANGE_DETECTORS + CHANGE_DETECTORS
