"""Test instances for change detectors in ``skchange.new_api.detectors``."""

from skchange.new_api.detectors._moving_window import MovingWindow
from skchange.new_api.interval_scorers import is_change_score, is_penalised_score
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

DETECTOR_TEST_INSTANCES = [
    MovingWindow(scorer)
    for scorer in INTERVAL_SCORER_TEST_INSTANCES
    if is_penalised_score(scorer) and is_change_score(scorer)
] + [
    MovingWindow(),
    MovingWindow(selection_method="detection_length"),
    MovingWindow(bandwidth=[2, 3, 5]),
]
