"""All changepoint detectors."""

from skchange.new_api.detectors._base import BaseChangeDetector, is_change_detector
from skchange.new_api.detectors._capa import CAPA
from skchange.new_api.detectors._moving_window import MovingWindow

__all__ = [
    "BaseChangeDetector",
    "CAPA",
    "MovingWindow",
    "is_change_detector",
]
