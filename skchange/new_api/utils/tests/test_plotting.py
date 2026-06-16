"""Tests for the new-API plotting utilities."""

import numpy as np
import pytest

plotly = pytest.importorskip("plotly")
import plotly.graph_objects as go  # noqa: E402

from skchange.new_api.utils import plot_detections, plot_segmentation  # noqa: E402
from skchange.new_api.utils.plotting import (  # noqa: E402
    _BASELINE_COLOR,
    _BASELINE_LABEL,
)


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


# ---------------------------------------------------------------------------
# Input-type coverage for _as_wide_array
# ---------------------------------------------------------------------------


class _FakeSeries:
    """Series-like stand-in: 1D values exposing a ``name`` attribute."""

    def __init__(self, values: np.ndarray, name: str) -> None:
        self._values = np.asarray(values)
        self.name = name

    def __array__(self, dtype: object = None) -> np.ndarray:
        return self._values if dtype is None else self._values.astype(dtype)


class _FakeFrame:
    """DataFrame-like stand-in: 2D values exposing a ``columns`` attribute."""

    def __init__(self, values: np.ndarray, columns: list[str]) -> None:
        self._values = np.asarray(values)
        self.columns = columns

    def __array__(self, dtype: object = None) -> np.ndarray:
        return self._values if dtype is None else self._values.astype(dtype)


def test_plot_detections_accepts_python_list_1d() -> None:
    fig = plot_detections([0.0, 1.0, 2.0, 3.0, 4.0], changepoints=np.array([2]))
    assert isinstance(fig, go.Figure)


def test_plot_detections_accepts_python_list_2d() -> None:
    data = [[i, i + 1.0] for i in range(10)]
    fig = plot_detections(data, changepoints=np.array([4]))
    assert isinstance(fig, go.Figure)


def test_plot_detections_accepts_dict_and_uses_keys_as_columns(
    rng: np.random.Generator,
) -> None:
    data = {"a": rng.normal(size=20), "b": rng.normal(size=20)}
    fig = plot_detections(data, changepoints=np.array([10]), data_repr="line")
    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"a", "b"}


def test_plot_detections_accepts_dataframe_like_and_uses_columns(
    rng: np.random.Generator,
) -> None:
    df = _FakeFrame(rng.normal(size=(30, 2)), columns=["x", "y"])
    fig = plot_detections(df, changepoints=np.array([15]), data_repr="line")
    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"x", "y"}


def test_plot_detections_accepts_series_like_and_uses_name(
    rng: np.random.Generator,
) -> None:
    series = _FakeSeries(rng.normal(size=25), name="signal")
    fig = plot_detections(series, changepoints=np.array([10]), data_repr="line")
    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"signal"}


def test_plot_detections_uses_default_value_label_for_unnamed_1d(
    rng: np.random.Generator,
) -> None:
    fig = plot_detections(
        rng.normal(size=20), changepoints=np.array([10]), data_repr="line"
    )
    trace_names = {trace.name for trace in fig.data}
    assert trace_names == {"value"}


def test_plot_detections_rejects_3d_input(rng: np.random.Generator) -> None:
    with pytest.raises(ValueError):
        plot_detections(rng.normal(size=(5, 2, 2)), changepoints=np.array([2]))


def test_plot_segmentation_accepts_dict_input(rng: np.random.Generator) -> None:
    data = {"a": rng.normal(size=30), "b": rng.normal(size=30)}
    fig = plot_segmentation(data, labels=np.repeat([0, 1], 15), x_var=0, y_var=1)
    assert fig.layout.xaxis.title.text == "a"
    assert fig.layout.yaxis.title.text == "b"
