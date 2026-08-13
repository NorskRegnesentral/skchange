"""Plotting utilities.

Requires plotly to be installed. Plotly is imported lazily inside the
plotting functions so importing this module (and therefore
``skchange.utils``) does not require it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from skchange.utils.segmentation import changepoints_to_labels

if TYPE_CHECKING:
    import plotly.graph_objects as go

_VALID_DATA_REPRS = ("line", "subplot-line", "point", "subplot-point", "heatmap")


def check_plotly_support(caller_name: str) -> None:
    """Raise ImportError with a detailed message if plotly is not installed.

    Plotting utilities should lazily import plotly and call this helper
    before any computation, mirroring scikit-learn's
    ``check_matplotlib_support``.

    Parameters
    ----------
    caller_name : str
        The name of the caller that requires plotly.
    """
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"{caller_name} requires plotly. "
            "You can install plotly with `pip install plotly`."
        ) from exc


def _as_wide_array(X: Any) -> tuple[np.ndarray, list[str]]:
    """Convert ``X`` to a wide 2D numpy array and a list of column names.

    Accepts the same kinds of inputs as ``plotly.express`` wide-form plotting:
    1D and 2D numpy arrays, Python lists, mappings (``dict``), pandas
    DataFrames (or any object with a ``columns`` attribute), and pandas Series
    (or any object with a ``name`` attribute). Pandas is not imported, so the
    helper relies on duck typing.

    Column names default to ``"value"`` for 1D inputs and ``"0"``, ``"1"``,
    ... for 2D arrays without explicit names.

    Parameters
    ----------
    X : array-like, mapping, DataFrame-like or Series-like
        Input data.

    Returns
    -------
    arr_2d : np.ndarray of shape (n_samples, n_features)
        Wide-form values with a positional integer row index.
    columns : list of str
        Column names, one per feature.
    """
    if isinstance(X, dict):
        columns = [str(k) for k in X]
        arr_2d = np.column_stack([np.asarray(v) for v in X.values()])
        return arr_2d, columns
    columns_attr = getattr(X, "columns", None)
    if columns_attr is not None:
        columns = [str(c) for c in list(columns_attr)]
        arr_2d = np.asarray(X)
        if arr_2d.ndim == 1:
            arr_2d = arr_2d.reshape(-1, 1)
        return arr_2d, columns
    arr = np.asarray(X)
    if arr.ndim == 1:
        name = getattr(X, "name", None)
        return arr.reshape(-1, 1), [str(name) if name is not None else "value"]
    if arr.ndim == 2:
        return arr, [str(i) for i in range(arr.shape[1])]
    raise ValueError(f"X must be 1D or 2D, got shape {arr.shape}.")


def _resolve_data_repr(
    data_repr: str | None,
    n_features: int,
    max_features_for_line_plot: int = 10,
) -> str:
    """Resolve ``data_repr=None`` to ``"heatmap"`` or ``"subplot-line"``."""
    if data_repr is None:
        return "heatmap" if n_features > max_features_for_line_plot else "subplot-line"
    if data_repr not in _VALID_DATA_REPRS:
        raise ValueError(
            f"Unknown data representation: {data_repr!r}. "
            f"Must be one of {_VALID_DATA_REPRS}."
        )
    return data_repr


def _plot_time_series(
    arr_2d: np.ndarray,
    columns: list[str],
    data_repr: str,
    **kwargs: Any,
) -> go.Figure:
    """Create a plotly figure from a wide 2D array and column names.

    Uses long-form keyword arrays so plotly does not require pandas.
    Column names are used as trace names (line, point) and as facet labels.
    """
    import plotly.express as px

    if data_repr == "heatmap":
        return px.imshow(
            arr_2d.T,
            aspect="auto",
            color_continuous_scale="Viridis",
            x=np.arange(arr_2d.shape[0]),
            y=columns,
            labels={"x": "index", "y": "feature", "color": "value"},
            **kwargs,
        )
    # go.Scatter has no dataframe backend dependency (unlike plotly.express).
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_samples, n_cols = arr_2d.shape
    x = np.arange(n_samples)
    mode = "lines" if "line" in data_repr else "markers"

    if data_repr.startswith("subplot-"):
        fig = make_subplots(rows=n_cols, cols=1, shared_xaxes=True)
        for i, name in enumerate(columns):
            fig.add_trace(
                go.Scatter(x=x, y=arr_2d[:, i], mode=mode, name=name),
                row=i + 1,
                col=1,
            )
            fig.update_yaxes(title_text=name, row=i + 1, col=1)
        fig.update_xaxes(title_text="index", row=n_cols, col=1)
        fig.update_layout(legend_title_text="feature")
    else:
        fig = go.Figure()
        for i, name in enumerate(columns):
            fig.add_trace(go.Scatter(x=x, y=arr_2d[:, i], mode=mode, name=name))
        fig.update_layout(yaxis_title="value", legend_title_text="feature")
    if kwargs:
        fig.update_layout(**kwargs)
    return fig


def plot_detections(
    X: ArrayLike,
    *,
    changepoints: ArrayLike | None = None,
    segment_anomalies: ArrayLike | None = None,
    affected_features: list[ArrayLike] | None = None,
    data_repr: str | None = None,
    **kwargs: Any,
) -> go.Figure:
    """Plot a time series with changepoints or segment anomalies overlaid.

    Provide exactly one of ``changepoints`` or ``segment_anomalies``.
    Changepoints are drawn as red dashed vertical lines. Segment anomalies are
    drawn as red translucent rectangles spanning ``[start, end)``.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        The time series data to plot.
    changepoints : array-like of shape (n_changepoints,), optional
        Changepoint indices, as returned by ``predict_changepoints``.
    segment_anomalies : array-like of shape (n_anomalies, 2), optional
        Segment-anomaly intervals as ``[start, end)`` pairs, as returned by
        ``predict_segment_anomalies``.
    affected_features : list of array-like, optional
        Per-anomaly arrays of feature indices that the anomaly affects, as
        returned by ``predict_all`` under ``"segment_anomaly_features"``. Only
        valid together with ``segment_anomalies`` and only honoured when the
        plot has one subplot per feature.
    data_repr : {"line", "subplot-line", "point", "subplot-point", "heatmap"}, optional
        Representation of the time series. If None, ``"heatmap"`` is used when
        there are more than 10 features, otherwise ``"subplot-line"``.
    **kwargs
        Forwarded to the underlying ``plotly.express`` plotting function.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with the time series and the detected events highlighted.
    """
    if (changepoints is None) == (segment_anomalies is None):
        raise ValueError(
            "Provide exactly one of `changepoints` or `segment_anomalies`."
        )
    if changepoints is not None and affected_features is not None:
        raise ValueError(
            "`affected_features` is only supported with `segment_anomalies`."
        )

    check_plotly_support("plot_detections")
    arr_2d, columns = _as_wide_array(X)
    n_vars = arr_2d.shape[1]
    data_repr = _resolve_data_repr(data_repr, n_vars)
    fig = _plot_time_series(arr_2d, columns, data_repr, **kwargs)
    n_subplots = n_vars if data_repr.startswith("subplot-") else 1
    visual_cpt_adjustment = -0.5 if data_repr == "heatmap" else 0.0

    if changepoints is not None:
        cps = np.asarray(changepoints, dtype=int).ravel()
        cols_per_event = [range(n_vars)] * len(cps) if n_subplots > 1 else None
        for i, cp in enumerate(cps):
            cols = cols_per_event[i] if n_subplots > 1 else [0]
            for col in cols:
                fig.add_vline(
                    x=float(cp) + visual_cpt_adjustment,
                    line_width=2,
                    line_dash="dash",
                    line_color="red",
                    row=col + 1,
                    col=1,
                )
        return fig

    anomalies = np.asarray(segment_anomalies, dtype=int)
    if anomalies.ndim != 2 or anomalies.shape[1] != 2:
        raise ValueError(
            "`segment_anomalies` must have shape (n_anomalies, 2), got "
            f"{anomalies.shape}."
        )
    if affected_features is not None and len(affected_features) != len(anomalies):
        raise ValueError(
            "`affected_features` must have one entry per row in `segment_anomalies`."
        )

    for i, (start, end) in enumerate(anomalies):
        if n_subplots > 1:
            if affected_features is not None:
                cols = np.asarray(affected_features[i], dtype=int).ravel()
            else:
                cols = range(n_vars)
        else:
            cols = [0]
        for col in cols:
            fig.add_vrect(
                x0=int(start),
                x1=int(end),
                fillcolor="red",
                opacity=0.2,
                line_width=0,
                layer="below",
                row=int(col) + 1,
                col=1,
            )
    return fig


_BASELINE_LABEL = "normal"
_BASELINE_COLOR = "black"


def _labels_to_interval_strings(labels: np.ndarray) -> np.ndarray:
    """Map each sample to a ``"[start, end)"`` string for its contiguous run."""
    n = len(labels)
    if n == 0:
        return np.array([], dtype=object)
    boundaries = np.concatenate(
        [[0], np.where(labels[1:] != labels[:-1])[0] + 1, [n]]
    ).astype(int)
    out = np.empty(n, dtype=object)
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        out[start:end] = f"[{start}, {end})"
    return out


def _segment_color_strings(
    n_samples: int,
    *,
    labels: ArrayLike | None,
    changepoints: ArrayLike | None,
    segment_anomalies: ArrayLike | None,
) -> tuple[np.ndarray, dict[str, str] | None]:
    """Build per-sample colour categories and an optional discrete-colour map.

    Exactly one of ``labels``, ``changepoints`` or ``segment_anomalies`` must
    be supplied. The category strings are used directly as plotly colour
    categories, and the returned map fixes specific categories to specific
    colours (used to render the ``"normal"`` baseline in black for
    segment-anomaly inputs).
    """
    n_given = sum(x is not None for x in (labels, changepoints, segment_anomalies))
    if n_given != 1:
        raise ValueError(
            "Provide exactly one of `labels`, `changepoints`, or `segment_anomalies`."
        )
    if labels is not None:
        arr = np.asarray(labels)
        if arr.shape != (n_samples,):
            raise ValueError(
                f"`labels` must have shape ({n_samples},), got {arr.shape}."
            )
        return arr.astype(str), None
    if changepoints is not None:
        cps = np.asarray(changepoints, dtype=int).ravel()
        dense = changepoints_to_labels(cps, n_samples)
        return _labels_to_interval_strings(dense), None
    anoms = np.asarray(segment_anomalies, dtype=int)
    if anoms.ndim != 2 or anoms.shape[1] != 2:
        raise ValueError(
            f"`segment_anomalies` must have shape (n_anomalies, 2), got {anoms.shape}."
        )
    out = np.full(n_samples, _BASELINE_LABEL, dtype=object)
    for s, e in anoms:
        s, e = int(s), int(e)
        out[s:e] = f"[{s}, {e})"
    return out, {_BASELINE_LABEL: _BASELINE_COLOR}


def plot_segmentation(
    X: ArrayLike,
    *,
    labels: ArrayLike | None = None,
    changepoints: ArrayLike | None = None,
    segment_anomalies: ArrayLike | None = None,
    x_var: int | str = "index",
    y_var: int | None = None,
) -> go.Figure:
    """Plot a segmentation of a time series.

    Provide exactly one of ``labels``, ``changepoints`` or
    ``segment_anomalies`` to define the segments. Each sample is rendered as
    a coloured point.

    The view depends on ``x_var``:

    * ``"index"`` (default) gives a time-series view with one subplot per
      feature, x is the sample index and y is the feature value.
    * an integer column index gives a feature-space scatter plot of feature
      ``x_var`` against feature ``y_var`` (which must be supplied).

    The colour encoding depends on which segmentation input is supplied:

    * ``labels``: one colour per distinct integer label (so non-contiguous
      runs of the same label share a colour).
    * ``changepoints``: one colour per contiguous run, labelled
      ``"[start, end)"``.
    * ``segment_anomalies``: one colour per anomaly interval, labelled
      ``"[start, end)"``. All baseline samples are labelled ``"normal"`` and
      rendered in black.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        The time series data.
    labels : array-like of shape (n_samples,), optional
        Per-sample segment labels, as returned by ``predict``.
    changepoints : array-like of shape (n_changepoints,), optional
        Changepoint indices, as returned by ``predict_changepoints``.
    segment_anomalies : array-like of shape (n_anomalies, 2), optional
        Segment-anomaly intervals as ``[start, end)`` pairs, as returned by
        ``predict_segment_anomalies``.
    x_var : int or "index", default="index"
        ``"index"`` selects a time-series view with one subplot per feature.
        An integer column index selects a feature-space scatter view, in
        which case ``y_var`` must also be supplied.
    y_var : int, optional
        Column index for the y-axis in scatter view. Ignored when
        ``x_var="index"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Figure with samples coloured by segment.
    """
    check_plotly_support("plot_segmentation")
    import plotly.colors
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    arr_2d, columns = _as_wide_array(X)
    n_samples, n_vars = arr_2d.shape
    segments, color_map = _segment_color_strings(
        n_samples,
        labels=labels,
        changepoints=changepoints,
        segment_anomalies=segment_anomalies,
    )

    if x_var == "index":
        unique_segs = list(dict.fromkeys(segments.tolist()))
        color_seq = plotly.colors.qualitative.Plotly
        resolved_cmap = color_map or {}
        seg_to_color = {
            seg: resolved_cmap.get(seg, color_seq[i % len(color_seq)])
            for i, seg in enumerate(unique_segs)
        }
        fig = make_subplots(rows=n_vars, cols=1, shared_xaxes=True)
        shown_in_legend: set[str] = set()
        for j, col_name in enumerate(columns):
            for seg in unique_segs:
                mask = segments == seg
                fig.add_trace(
                    go.Scatter(
                        x=np.where(mask)[0],
                        y=arr_2d[mask, j],
                        mode="markers",
                        name=seg,
                        marker_color=seg_to_color[seg],
                        showlegend=seg not in shown_in_legend,
                        legendgroup=seg,
                    ),
                    row=j + 1,
                    col=1,
                )
                shown_in_legend.add(seg)
            fig.update_yaxes(title_text=col_name, row=j + 1, col=1)
        fig.update_xaxes(title_text="index", row=n_vars, col=1)
        fig.update_layout(legend_title_text="segment")
        return fig

    if y_var is None:
        raise ValueError("`y_var` is required when `x_var` is a column index.")
    if n_vars < 2:
        raise ValueError(
            f"Scatter view requires at least two features, got shape {arr_2d.shape}."
        )
    x_idx, y_idx = int(x_var), int(y_var)
    x_name, y_name = columns[x_idx], columns[y_idx]
    if x_name == y_name:
        x_name, y_name = f"{x_name} (x)", f"{y_name} (y)"
    unique_segs = list(dict.fromkeys(segments.tolist()))
    color_seq = plotly.colors.qualitative.Plotly
    resolved_cmap = color_map or {}
    seg_to_color = {
        seg: resolved_cmap.get(seg, color_seq[i % len(color_seq)])
        for i, seg in enumerate(unique_segs)
    }
    fig = go.Figure()
    for seg in unique_segs:
        mask = segments == seg
        fig.add_trace(
            go.Scatter(
                x=arr_2d[mask, x_idx],
                y=arr_2d[mask, y_idx],
                mode="markers",
                name=seg,
                marker_color=seg_to_color[seg],
            )
        )
    fig.update_layout(
        xaxis_title=x_name, yaxis_title=y_name, legend_title_text="segment"
    )
    return fig
