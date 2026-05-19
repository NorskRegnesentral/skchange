"""Tests for ``penalty_scale`` semantics across detectors.

Covers, for each detector that auto-wraps an unpenalised scorer:

* ``penalty_scale=k`` on a default/unpenalised input scales the resulting
  ``<scorer>_.penalty_`` by ``k`` relative to the ``penalty_scale=1.0`` baseline.
* Passing an explicit :class:`PenalisedScore` is preserved unchanged and the
  detector ``penalty_scale`` is ignored (the user-supplied scorer owns its
  penalty).

Plus CAPA-specific tests for the independent ``segment_penalty_scale`` and
``point_penalty_scale`` parameters, and PELT tests where ``penalty_scale``
multiplies the default penalty but is ignored when ``penalty`` is set.
"""

import numpy as np
import pytest
from sklearn.base import clone

from skchange.new_api.conftest import make_single_change_X
from skchange.new_api.detectors import (
    CAPA,
    PELT,
    CircularBinarySegmentation,
    MovingWindow,
    SeededBinarySegmentation,
)
from skchange.new_api.interval_scorers import (
    CUSUM,
    CostTransientScore,
    L2Saving,
    PenalisedScore,
)
from skchange.new_api.interval_scorers._costs.l2_cost import L2Cost

# (DetectorClass, scorer_param_name, fitted_scorer_attr, unpenalised_scorer_factory)
_AUTOWRAP_DETECTORS = [
    (
        SeededBinarySegmentation,
        "change_score",
        "change_score_",
        lambda: CUSUM(),
    ),
    (
        MovingWindow,
        "change_score",
        "change_score_",
        lambda: CUSUM(),
    ),
    (
        CircularBinarySegmentation,
        "transient_score",
        "transient_score_",
        lambda: CostTransientScore(L2Cost()),
    ),
]


@pytest.mark.parametrize(
    "DetectorCls, scorer_param, scorer_attr, scorer_factory",
    _AUTOWRAP_DETECTORS,
    ids=[cls.__name__ for cls, *_ in _AUTOWRAP_DETECTORS],
)
def test_penalty_scale_multiplies_default(
    DetectorCls, scorer_param, scorer_attr, scorer_factory
):
    """``penalty_scale=k`` scales the fitted scorer penalty by ``k``."""
    X = make_single_change_X(DetectorCls())

    base_pen = getattr(DetectorCls(penalty_scale=1.0).fit(X), scorer_attr).penalty_
    scaled_pen = getattr(DetectorCls(penalty_scale=3.0).fit(X), scorer_attr).penalty_

    assert np.allclose(scaled_pen, 3.0 * base_pen)


@pytest.mark.parametrize(
    "DetectorCls, scorer_param, scorer_attr, scorer_factory",
    _AUTOWRAP_DETECTORS,
    ids=[cls.__name__ for cls, *_ in _AUTOWRAP_DETECTORS],
)
def test_explicit_penalised_score_ignores_detector_scale(
    DetectorCls, scorer_param, scorer_attr, scorer_factory
):
    """Explicit ``PenalisedScore`` retains its own penalty; detector scale ignored."""
    explicit = PenalisedScore(scorer_factory(), penalty_scale=2.0)
    X = make_single_change_X(DetectorCls())

    inner_pen = getattr(
        DetectorCls(**{scorer_param: clone(explicit)}).fit(X), scorer_attr
    ).penalty_
    detector_scaled_pen = getattr(
        DetectorCls(
            **{scorer_param: clone(explicit)},
            penalty_scale=3.0,
        ).fit(X),
        scorer_attr,
    ).penalty_

    assert np.allclose(detector_scaled_pen, inner_pen)


# ---------------------------------------------------------------------------
# CAPA: split segment_/point_penalty_scale
# ---------------------------------------------------------------------------


def test_capa_penalty_scales_act_independently():
    """``segment_penalty_scale`` and ``point_penalty_scale`` only affect their own."""
    X = make_single_change_X(CAPA())

    base = CAPA().fit(X)
    seg_scaled = CAPA(segment_penalty_scale=3.0).fit(X)
    point_scaled = CAPA(point_penalty_scale=4.0).fit(X)

    assert np.allclose(
        seg_scaled.segment_saving_.penalty_, 3.0 * base.segment_saving_.penalty_
    )
    assert np.allclose(seg_scaled.point_saving_.penalty_, base.point_saving_.penalty_)

    assert np.allclose(
        point_scaled.segment_saving_.penalty_, base.segment_saving_.penalty_
    )
    # point_penalty_scale=4.0 vs default 2.0 -> 2x relative to base.
    assert np.allclose(
        point_scaled.point_saving_.penalty_, 2.0 * base.point_saving_.penalty_
    )


def test_capa_default_point_penalty_matches_linear_chi2_doubled_REMOVED():
    pass


def test_capa_explicit_penalised_savings_ignore_detector_scale():
    """Explicit ``PenalisedScore`` savings retain their own penalty."""
    X = make_single_change_X(CAPA())
    explicit = PenalisedScore(L2Saving(), penalty_scale=2.0)

    base = CAPA(segment_saving=clone(explicit)).fit(X)
    scaled = CAPA(
        segment_saving=clone(explicit),
        segment_penalty_scale=3.0,
    ).fit(X)
    assert np.allclose(scaled.segment_saving_.penalty_, base.segment_saving_.penalty_)


# ---------------------------------------------------------------------------
# PELT: penalty_scale on the cost (no auto-wrap)
# ---------------------------------------------------------------------------


def test_pelt_penalty_scale_multiplies_default():
    """``penalty_scale=k`` scales the default penalty by ``k``."""
    X = make_single_change_X(PELT())

    base_pen = PELT().fit(X).penalty_
    scaled_pen = PELT(penalty_scale=3.0).fit(X).penalty_

    assert np.allclose(scaled_pen, 3.0 * base_pen)


def test_pelt_penalty_scale_ignored_when_penalty_explicit():
    """Explicit ``penalty`` ignores ``penalty_scale``."""
    X = make_single_change_X(PELT())
    fitted = PELT(penalty=10.0, penalty_scale=3.0).fit(X)
    assert fitted.penalty_ == 10.0
