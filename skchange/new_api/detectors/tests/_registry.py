"""Test instances for change detectors in ``skchange.new_api.detectors``."""

from skchange.new_api.detectors import (
    CAPA,
    CROPS,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.interval_scorers import (
    is_change_score,
    is_cost,
    is_penalised_score,
    is_saving,
    is_transient_score,
)
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

_MOVING_WINDOW_INSTANCES = [
    MovingWindow(),
    MovingWindow(selection_method="detection_length", bandwidth=5),
    MovingWindow(bandwidth=5),
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
        CAPA(segment_saving=scorer, include_point_anomalies=True)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_saving(scorer)
    ],
]

_PELT_INSTANCES = [
    PELT(),
    PELT(min_segment_length=5),
    PELT(penalty=10.0),
    PELT(step_size=5),
    *[
        PELT(cost=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if not is_penalised_score(scorer) and is_cost(scorer)
    ],
]

_SEEDED_BINSEG_INSTANCES = [
    SeededBinarySegmentation(),
    SeededBinarySegmentation(max_interval_length=100),
    SeededBinarySegmentation(selection_method="narrowest"),
    *[
        SeededBinarySegmentation(change_score=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_change_score(scorer)
    ],
]

_CROPS_INSTANCES = [
    CROPS(),
    CROPS(min_penalty=0.5, max_penalty=20.0, min_segment_length=5),
    CROPS(selection_method="elbow"),
    CROPS(step_size=5),
    *[
        CROPS(cost=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if not is_penalised_score(scorer) and is_cost(scorer)
    ],
]

_CIRCULAR_BINSEG_INSTANCES = [
    CircularBinarySegmentation(),
    CircularBinarySegmentation(min_subinterval_length=10, max_interval_length=50),
    *[
        CircularBinarySegmentation(transient_score=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_penalised_score(scorer) and is_transient_score(scorer)
    ],
]

DETECTOR_TEST_INSTANCES = [
    *_MOVING_WINDOW_INSTANCES,
    *_CAPA_INSTANCES,
    *_PELT_INSTANCES,
    *_SEEDED_BINSEG_INSTANCES,
    *_CROPS_INSTANCES,
    *_CIRCULAR_BINSEG_INSTANCES,
]
