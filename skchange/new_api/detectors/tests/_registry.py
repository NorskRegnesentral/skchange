"""Test instances for change detectors in ``skchange.new_api.detectors``."""

from skchange.new_api.detectors._capa import CAPA
from skchange.new_api.detectors._moving_window import MovingWindow
from skchange.new_api.detectors._pelt import PELT
from skchange.new_api.interval_scorers import (
    is_change_score,
    is_cost,
    is_penalised_score,
    is_saving,
)
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

_MOVING_WINDOW_INSTANCES = [
    MovingWindow(),
    MovingWindow(selection_method="detection_length"),
    MovingWindow(bandwidth=[2, 3, 5]),
    *[
        MovingWindow(scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_change_score(scorer)
    ],
]

_CAPA_INSTANCES = [
    CAPA(),
    CAPA(min_segment_length=10, max_segment_length=100),
    *[
        CAPA(segment_saving=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_saving(scorer)
    ],
    *[
        CAPA(segment_saving=scorer, point_saving=scorer, include_point_anomalies=True)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_saving(scorer)
    ],
]

_PELT_INSTANCES = [
    PELT(),
    PELT(min_segment_length=5),
    PELT(penalty=10.0),
    PELT(step_size=2),
    *[
        PELT(cost=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if not is_penalised_score(scorer) and is_cost(scorer)
    ],
]

DETECTOR_TEST_INSTANCES = [
    *_MOVING_WINDOW_INSTANCES,
    *_CAPA_INSTANCES,
    *_PELT_INSTANCES,
]
