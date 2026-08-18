"""All changepoint detectors."""

from skchange.detectors._base import BaseChangeDetector, is_change_detector
from skchange.detectors._capa import CAPA
from skchange.detectors._circular_binseg import CircularBinarySegmentation
from skchange.detectors._crops import CROPS
from skchange.detectors._fpop import FPOP
from skchange.detectors._moving_window import MovingWindow
from skchange.detectors._pelt import PELT
from skchange.detectors._seeded_binseg import SeededBinarySegmentation

__all__ = [
    "BaseChangeDetector",
    "CAPA",
    "CircularBinarySegmentation",
    "CROPS",
    "FPOP",
    "MovingWindow",
    "PELT",
    "SeededBinarySegmentation",
    "is_change_detector",
]
