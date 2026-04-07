"""All changepoint detectors."""

from skchange.new_api.detectors._base import BaseChangeDetector, is_change_detector
from skchange.new_api.detectors._capa import CAPA
from skchange.new_api.detectors._moving_window import MovingWindow
from skchange.new_api.detectors._pelt import PELT

__all__ = [
    "BaseChangeDetector",
    "CAPA",
    "MovingWindow",
    "PELT",
    "is_change_detector",
]
