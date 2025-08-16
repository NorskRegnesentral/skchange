"""Plotting utilities."""

import pandas as pd
import plotly.express as px


def plot_change_points(df: pd.DataFrame, change_points: pd.DataFrame):
    """Plot change points on a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to plot. The index should represent the time points, while
        the columns represent the values of the time series.

    change_points : pd.DataFrame
        The change points to plot, on the format returned by detectors' `predict`
        method. I.e., it must contain an `"ilocs"` column with the integer indices of
        the change points.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with the time series and change points highlighted by
        vertical red, dashed lines.
    """
    df = df.copy()
    fig = px.line(df)
    for cpt in change_points["ilocs"]:
        fig.add_vline(x=cpt, line_width=2, line_dash="dash", line_color="red")
    return fig


def plot_scatter_segmentation(
    df: pd.DataFrame, segment_labels: pd.DataFrame, x_var=None, y_var=None
):
    """Plot segmentation of a time series.

    Parameters
    ----------
    df : pd.DataFrame
        The time series data to plot. The index should represent the time points, while
        the columns represent the values of the time series.

    segment_labels : pd.DataFrame
        The segment labels, on the format returned by detectors' `predict` method.
        It must contain a `"labels"` column with the segment labels.

    x_var : str, optional
        The name of the column to use for the x-axis. If None, the index will be used.

    y_var : str, optional
        The name of the column to use for the y-axis. If None, the first column will be
        used.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with the time series and segments highlighted by different
        colors.
    """
    df = df.copy()
    original_columns = df.columns.tolist()

    # Find first and last index per group
    segment_ranges = (
        segment_labels.reset_index()
        .groupby("labels")
        .agg(start_inclusive=("index", "first"), end_inclusive=("index", "last"))
        .reset_index()
    )
    segment_ranges["end_exclusive"] = segment_ranges["end_inclusive"] + 1

    df["labels"] = segment_labels["labels"]
    df = df.join(
        segment_ranges.set_index("labels")[["start_inclusive", "end_exclusive"]],
        on="labels",
    )
    df["segment"] = df.apply(
        lambda row: f"[{int(row['start_inclusive'])}, {int(row['end_exclusive'])})",
        axis=1,
    )
    df = df[original_columns + ["segment"]]

    fig = px.scatter(df, x=x_var, y=y_var, color="segment")
    return fig
