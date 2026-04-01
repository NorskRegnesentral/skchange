"""Test instances for change detectors in ``skchange.new_api.detectors``."""

from skchange.new_api.detectors._moving_window import MovingWindow
from skchange.new_api.interval_scorers.tests._registry import (
    CHANGE_SCORE_TEST_INSTANCES,
)

DETECTOR_TEST_INSTANCES = [
    (MovingWindow, {"change_score": cls(**params)})
    for cls, params in CHANGE_SCORE_TEST_INSTANCES
] + [
    (MovingWindow, {}),
    (MovingWindow, {"bandwidth": 5}),
    (MovingWindow, {"selection_method": "detection_length", "bandwidth": 5}),
]
