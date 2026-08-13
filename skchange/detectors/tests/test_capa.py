"""Dedicated tests for `CAPA` paths not covered by the common contract suite."""

import numpy as np
import pytest

from skchange.detectors import CAPA
from skchange.interval_scorers import L2Cost, MultivariateGaussianSaving

TRUE_POINT_ANOMALY = 20


def _make_capa_X(seed: int = 0) -> np.ndarray:
    """Generate data with an obvious segment anomaly and two obvious point anomalies."""
    rng = np.random.default_rng(seed)
    n_samples, n_features = 200, 3
    X = rng.standard_normal((n_samples, n_features))
    # Segment anomaly in features 0 and 1.
    X[80:100, 0] += 6.0
    X[80:100, 1] += 6.0
    # Two point anomalies far from the segment.
    X[TRUE_POINT_ANOMALY, 2] += 15.0
    return X


def test_capa_include_point_anomalies_detects_point_anomalies():
    """With include_point_anomalies=True, predict_all must surface point anomalies."""
    X = _make_capa_X()
    detector = CAPA(include_point_anomalies=True).fit(X)
    result = detector.predict_all(X)

    point_anomalies = result["point_anomalies"]
    assert isinstance(point_anomalies, np.ndarray)
    assert point_anomalies.ndim == 1
    assert (
        len(point_anomalies) >= 1
    ), f"Expected at least one detected point anomaly, got {point_anomalies}"
    assert np.any(
        np.abs(point_anomalies - TRUE_POINT_ANOMALY) <= 1
    ), f"No detected point anomaly near {TRUE_POINT_ANOMALY}, got {point_anomalies}"

    intervals = detector.predict_segment_anomalies(X)
    assert isinstance(intervals, np.ndarray)
    assert intervals.ndim == 2 and intervals.shape[1] == 2
    # Segment anomaly [80, 100) should be detected.
    assert np.any(
        (intervals[:, 0] <= 80) & (intervals[:, 1] >= 100)
    ), f"Segment anomaly [80, 100) not found in {intervals}"
    # Point anomaly near sample 20 should appear as a single-sample interval.
    point_intervals = intervals[intervals[:, 1] - intervals[:, 0] == 1]
    assert np.any(
        np.abs(point_intervals[:, 0] - TRUE_POINT_ANOMALY) <= 1
    ), f"No single-sample interval near {TRUE_POINT_ANOMALY} in {intervals}"


@pytest.mark.parametrize("include_point_anomalies", [False, True])
def test_capa_predict_scores_return_index(include_point_anomalies):
    """predict_scores(return_index=True) must return aligned scores/starts/ends,
    with point intervals concatenated when ``include_point_anomalies=True``.
    """
    X = _make_capa_X()
    detector = CAPA(include_point_anomalies=include_point_anomalies).fit(X)

    scores, index = detector.predict_scores(X, return_index=True)

    assert isinstance(scores, np.ndarray)
    assert scores.ndim == 1
    assert set(index.keys()) == {"starts", "ends"}
    starts = index["starts"]
    ends = index["ends"]
    assert starts.shape == scores.shape
    assert ends.shape == scores.shape
    assert np.all(ends > starts)


def test_capa_fit_rejects_non_saving_segment_scorer():
    with pytest.raises(ValueError, match="segment_saving"):
        CAPA(segment_saving=L2Cost()).fit(_make_capa_X())


def test_capa_fit_rejects_non_saving_point_scorer():
    with pytest.raises(ValueError, match="point_saving"):
        CAPA(point_saving=L2Cost()).fit(_make_capa_X())


def test_capa_fit_rejects_point_saving_with_min_size_gt_1():
    # MultivariateGaussianSaving has min_size = n_features + 1 = 4 after fit.
    with pytest.raises(ValueError, match="min_size"):
        CAPA(point_saving=MultivariateGaussianSaving()).fit(_make_capa_X())


def test_capa_fit_rejects_min_segment_length_below_saving_min_size():
    # MultivariateGaussianSaving has min_size = 4; min_segment_length=2 < 4.
    with pytest.raises(ValueError, match="min_segment_length"):
        CAPA(segment_saving=MultivariateGaussianSaving(), min_segment_length=2).fit(
            _make_capa_X()
        )


def test_capa_fit_rejects_min_segment_length_exceeding_max():
    with pytest.raises(ValueError, match="min_segment_length"):
        CAPA(min_segment_length=50, max_segment_length=10).fit(_make_capa_X())
