"""Test instances for change detectors in ``skchange.detectors``."""

from skchange.detectors import (
    CAPA,
    CROPS,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.interval_scorers import (
    CostTransientScore,
    GaussianCost,
    L1Saving,
    L2Cost,
    is_change_score,
    is_cost,
    is_penalised_score,
    is_saving,
)
from skchange.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

_MOVING_WINDOW_INSTANCES = [
    MovingWindow(),
    MovingWindow(selection_method="detection_length", bandwidth=5),
    MovingWindow(bandwidth=5),
    *[
        MovingWindow(scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_change_score(scorer)
    ],
]

_CAPA_INSTANCES = [
    CAPA(),
    CAPA(min_segment_length=10, max_segment_length=100),
    CAPA(segment_saving=L1Saving(), point_saving=L1Saving(), penalty_scale=2.0),
    *[
        CAPA(segment_saving=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_saving(scorer)
    ],
    *[
        CAPA(segment_saving=scorer, include_point_anomalies=True)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_saving(scorer)
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
    SeededBinarySegmentation(penalty=10.0),
    SeededBinarySegmentation(penalty_scale=2.0),
    SeededBinarySegmentation(agg="max"),
    SeededBinarySegmentation(growth_factor=2.0),
    *[
        SeededBinarySegmentation(change_score=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_change_score(scorer)
    ],
]

_CROPS_INSTANCES = [
    # CROPS wraps PELT; cost-coverage is handled by _PELT_INSTANCES.
    CROPS(),
    CROPS(min_penalty=0.5, max_penalty=20.0, min_segment_length=5),
    CROPS(selection_method="elbow"),
    CROPS(step_size=5),
]

_CIRCULAR_BINSEG_INSTANCES = [
    # CBS evaluates the transient score on a huge number of candidate
    # ``(outer, inner)`` interval pairs, so we test only a small representative
    # subset of transient scores to keep CI time reasonable.
    CircularBinarySegmentation(),
    CircularBinarySegmentation(min_subinterval_length=5, max_interval_length=100),
    CircularBinarySegmentation(penalty=10.0),
    CircularBinarySegmentation(penalty_scale=3.0),
    CircularBinarySegmentation(agg="max"),
    CircularBinarySegmentation(transient_score=CostTransientScore(L2Cost())),
    CircularBinarySegmentation(transient_score=CostTransientScore(GaussianCost())),
]

DETECTOR_TEST_INSTANCES = [
    *_MOVING_WINDOW_INSTANCES,
    *_CAPA_INSTANCES,
    *_PELT_INSTANCES,
    *_SEEDED_BINSEG_INSTANCES,
    *_CROPS_INSTANCES,
    *_CIRCULAR_BINSEG_INSTANCES,
]
