"""Tests for ``skchange.new_api.metrics._scorer``."""

import numpy as np
import pytest

from skchange.new_api.metrics._scorer import (
    BUILTIN_SCORERS,
    make_detector_scorer,
    n_segments,
    resolve_scoring,
)

# ---------------------------------------------------------------------------
# Built-in scorer: n_segments (counts contiguous label runs)
# ---------------------------------------------------------------------------


class _LabelsDetector:
    """Minimal stand-in that returns a fixed label array from ``predict``."""

    def __init__(self, labels):
        self._labels = np.asarray(labels)

    def predict(self, X):
        return self._labels


@pytest.mark.parametrize(
    "labels, expected",
    [
        ([], 0.0),
        ([0], 1.0),
        ([0, 0, 0], 1.0),
        ([0, 1, 1, 2], 3.0),
        ([0, 1, 0, 1, 0], 5.0),
    ],
)
def test_n_segments_counts_runs(labels, expected):
    """``n_segments`` counts contiguous label runs (k changepoints → k+1)."""
    detector = _LabelsDetector(labels)
    assert n_segments(detector, X=None) == expected


# ---------------------------------------------------------------------------
# resolve_scoring
# ---------------------------------------------------------------------------


def test_resolve_scoring_returns_builtin_for_known_name():
    """Known names resolve to the corresponding callable in ``BUILTIN_SCORERS``."""
    for name, expected in BUILTIN_SCORERS.items():
        assert resolve_scoring(name) is expected


def test_resolve_scoring_passes_callable_through():
    """A callable input is returned unchanged."""

    def my_scorer(detector, X, y=None):
        return 1.0

    assert resolve_scoring(my_scorer) is my_scorer


def test_resolve_scoring_unknown_name_raises():
    """Unknown built-in names are rejected."""
    with pytest.raises(ValueError):
        resolve_scoring("not_a_real_scorer")


# ---------------------------------------------------------------------------
# make_detector_scorer
# ---------------------------------------------------------------------------


class _PredictDetector:
    """Stand-in exposing the response methods used by ``make_detector_scorer``."""

    def __init__(self, *, changepoints=None, segment_anomalies=None):
        self._changepoints = (
            np.asarray(changepoints) if changepoints is not None else None
        )
        self._segment_anomalies = (
            np.asarray(segment_anomalies) if segment_anomalies is not None else None
        )

    def predict_changepoints(self, X):
        return self._changepoints

    def predict_segment_anomalies(self, X):
        return self._segment_anomalies


def test_make_detector_scorer_forwards_y_true_and_y_pred_to_metric():
    """The wrapped scorer calls ``metric(y, y_pred)`` with the y supplied at
    scoring time and y_pred obtained from the detector's response method.
    """
    received = {}

    def fake_metric(y_true, y_pred):
        received["y_true"] = y_true
        received["y_pred"] = y_pred
        return 0.5

    y_true = np.array([10, 50])
    y_pred = np.array([12, 48])
    detector = _PredictDetector(changepoints=y_pred)

    scorer = make_detector_scorer(fake_metric)
    score = scorer(detector, X=None, y=y_true)

    assert score == 0.5
    assert received["y_true"] is y_true
    np.testing.assert_array_equal(received["y_pred"], y_pred)


def test_make_detector_scorer_uses_response_method_argument():
    """``response_method`` selects which detector prediction method is called."""
    y_pred = np.array([[5, 10], [20, 30]])
    detector = _PredictDetector(segment_anomalies=y_pred)

    def identity_metric(y_true, y_pred):
        return float(np.array_equal(y_true, y_pred))

    scorer = make_detector_scorer(
        identity_metric, response_method="predict_segment_anomalies"
    )

    assert scorer(detector, X=None, y=y_pred) == 1.0


def test_make_detector_scorer_raises_when_y_is_none():
    """Metric-based scorers require a reference ``y``."""

    def fake_metric(y_true, y_pred):
        return 0.0

    scorer = make_detector_scorer(fake_metric)
    detector = _PredictDetector(changepoints=np.array([1, 2]))

    with pytest.raises(ValueError):
        scorer(detector, X=None, y=None)
