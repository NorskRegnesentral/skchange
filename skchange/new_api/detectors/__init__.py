"""All changepoint detectors."""

from skchange.new_api.detectors._base import BaseChangeDetector, is_change_detector
from skchange.new_api.detectors._capa import CAPA
from skchange.new_api.detectors._circular_binseg import CircularBinarySegmentation
from skchange.new_api.detectors._crops import CROPS
from skchange.new_api.detectors._moving_window import MovingWindow
from skchange.new_api.detectors._pelt import PELT
from skchange.new_api.detectors._seeded_binseg import SeededBinarySegmentation

__all__ = [
    "BaseChangeDetector",
    "CAPA",
    "CircularBinarySegmentation",
    "CROPS",
    "MovingWindow",
    "PELT",
    "SeededBinarySegmentation",
    "is_change_detector",
]
