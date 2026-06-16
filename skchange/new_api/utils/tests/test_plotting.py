"""Tests for the new-API plotting utilities."""

import numpy as np
import plotly.graph_objects as go
import pytest

from skchange.new_api.utils import plot_detections, plot_segmentation
from skchange.new_api.utils.plotting import _BASELINE_COLOR, _BASELINE_LABEL


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


@pytest.fixture
def X_1d(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=100)


@pytest.fixture
def X_2d(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(100, 3))


@pytest.fixture
def X_many(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(50, 12))


# ---------------------------------------------------------------------------
# plot_detections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "data_repr", ["line", "subplot-line", "point", "subplot-point", "heatmap"]
)
def test_plot_detections_supports_all_data_reprs(
    X_2d: np.ndarray, data_repr: str
) -> None:
    fig = plot_detections(X_2d, changepoints=np.array([25, 70]), data_repr=data_repr)
    assert isinstance(fig, go.Figure)


def test_plot_detections_with_1d_input(X_1d: np.ndarray) -> None:
    fig = plot_detections(X_1d, changepoints=np.array([40]))
    assert isinstance(fig, go.Figure)


def test_plot_detections_with_segment_anomalies(X_2d: np.ndarray) -> None:
    fig = plot_detections(X_2d, segment_anomalies=np.array([[10, 20], [60, 80]]))
    assert isinstance(fig, go.Figure)


def test_plot_detections_with_affected_features(X_2d: np.ndarray) -> None:
    fig = plot_detections(
        X_2d,
        segment_anomalies=np.array([[10, 20], [60, 80]]),
        affected_features=[np.array([0, 2]), np.array([1])],
        data_repr="subplot-line",
    )
    assert isinstance(fig, go.Figure)


def test_plot_detections_default_repr_is_heatmap_for_many_features(
    X_many: np.ndarray,
) -> None:
    fig = plot_detections(X_many, changepoints=np.array([10, 30]))
    assert any(trace.type == "heatmap" for trace in fig.data)


def test_plot_detections_rejects_zero_or_two_detection_kinds(
    X_2d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_detections(X_2d)
    with pytest.raises(ValueError):
        plot_detections(
            X_2d,
            changepoints=np.array([10]),
            segment_anomalies=np.array([[20, 30]]),
        )


def test_plot_detections_rejects_affected_features_without_anomalies(
    X_2d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_detections(
            X_2d,
            changepoints=np.array([10]),
            affected_features=[np.array([0])],
        )


def test_plot_detections_rejects_bad_anomaly_shape(X_2d: np.ndarray) -> None:
    with pytest.raises(ValueError):
        plot_detections(X_2d, segment_anomalies=np.array([10, 20, 30]))


def test_plot_detections_rejects_affected_features_length_mismatch(
    X_2d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_detections(
            X_2d,
            segment_anomalies=np.array([[10, 20], [60, 80]]),
            affected_features=[np.array([0])],
        )


def test_plot_detections_rejects_unknown_data_repr(X_2d: np.ndarray) -> None:
    with pytest.raises(ValueError):
        plot_detections(X_2d, changepoints=np.array([10]), data_repr="bogus")


# ---------------------------------------------------------------------------
# plot_segmentation
# ---------------------------------------------------------------------------


def test_plot_segmentation_time_view_with_labels_shares_color_per_label(
    X_2d: np.ndarray,
) -> None:
    labels = np.repeat([0, 1, 0], [40, 30, 30])
    fig = plot_segmentation(X_2d, labels=labels)
    # Non-contiguous samples with label 0 share the same colour category.
    assert {t.legendgroup for t in fig.data} == {"0", "1"}


def test_plot_segmentation_time_view_with_changepoints_colors_by_interval(
    X_2d: np.ndarray,
) -> None:
    fig = plot_segmentation(X_2d, changepoints=np.array([40, 70]))
    assert {t.legendgroup for t in fig.data} == {"[0, 40)", "[40, 70)", "[70, 100)"}


def test_plot_segmentation_time_view_with_segment_anomalies_uses_black_baseline(
    X_2d: np.ndarray,
) -> None:
    fig = plot_segmentation(X_2d, segment_anomalies=np.array([[10, 30], [60, 80]]))
    assert {t.legendgroup for t in fig.data} == {
        _BASELINE_LABEL,
        "[10, 30)",
        "[60, 80)",
    }
    baseline = next(t for t in fig.data if t.legendgroup == _BASELINE_LABEL)
    assert baseline.marker.color == _BASELINE_COLOR


def test_plot_segmentation_scatter_view_uses_columns_as_axis_titles(
    X_2d: np.ndarray,
) -> None:
    labels = np.repeat([0, 1, 2], [40, 30, 30])
    fig = plot_segmentation(X_2d, labels=labels, x_var=0, y_var=1)
    assert fig.layout.xaxis.title.text == "0"
    assert fig.layout.yaxis.title.text == "1"


def test_plot_segmentation_scatter_view_rejects_missing_y_var(
    X_2d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_segmentation(X_2d, changepoints=np.array([50]), x_var=0)


def test_plot_segmentation_scatter_view_rejects_single_feature_input(
    X_1d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_segmentation(X_1d, changepoints=np.array([40]), x_var=0, y_var=0)


def test_plot_segmentation_rejects_zero_or_two_segmentation_kinds(
    X_2d: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        plot_segmentation(X_2d)
    with pytest.raises(ValueError):
        plot_segmentation(
            X_2d, labels=np.zeros(100, dtype=int), changepoints=np.array([50])
        )


def test_plot_segmentation_rejects_label_length_mismatch(X_2d: np.ndarray) -> None:
    with pytest.raises(ValueError):
        plot_segmentation(X_2d, labels=np.zeros(50, dtype=int))


def test_plot_segmentation_rejects_bad_anomaly_shape(X_2d: np.ndarray) -> None:
    with pytest.raises(ValueError):
        plot_segmentation(X_2d, segment_anomalies=np.array([10, 20, 30]))
