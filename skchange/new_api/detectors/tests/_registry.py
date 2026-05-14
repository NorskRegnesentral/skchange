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
    CostTransientScore,
    GaussianCost,
    L2Cost,
    PenalisedScore,
    is_change_score,
    is_cost,
    is_penalised_score,
    is_saving,
)
from skchange.new_api.interval_scorers.tests._registry import (
    INTERVAL_SCORER_TEST_INSTANCES,
)

# ``PenalisedScore``-wrapped scorers are excluded from the registry-driven
# detector instances below: detectors auto-wrap unpenalised scorers in
# ``PenalisedScore`` internally, so passing a ``PenalisedScore(inner)`` would
# duplicate coverage of the ``inner`` scorer with no extra branches exercised.
# Inherently penalised scorers (e.g. ``ESACScore``) are still included to cover
# the "user supplies an already-penalised, non-``PenalisedScore`` scorer" path.

_MOVING_WINDOW_INSTANCES = [
    MovingWindow(),
    MovingWindow(selection_method="detection_length", bandwidth=5),
    MovingWindow(bandwidth=5),
    *[
        MovingWindow(scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_change_score(scorer) and not isinstance(scorer, PenalisedScore)
    ],
]

_CAPA_INSTANCES = [
    CAPA(),
    CAPA(min_segment_length=10, max_segment_length=100),
    *[
        CAPA(segment_saving=scorer)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_saving(scorer) and not isinstance(scorer, PenalisedScore)
    ],
    *[
        CAPA(segment_saving=scorer, include_point_anomalies=True)
        for scorer in INTERVAL_SCORER_TEST_INSTANCES
        if is_saving(scorer) and not isinstance(scorer, PenalisedScore)
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
        if is_change_score(scorer) and not isinstance(scorer, PenalisedScore)
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
    CircularBinarySegmentation(max_interval_length=50),
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
